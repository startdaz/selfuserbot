import asyncio
import contextlib
import html
import io
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s: %(message)s")
logging.getLogger("pyrogram").setLevel(logging.ERROR)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-U", "pip"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-U", "-r", "requirements.txt"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

import aiohttp
import aiorun
import pyrogram
from meval import meval
from pyrogram import Client, filters
from pyrogram.enums import ChatType, ParseMode
from pyrogram.handlers import DeletedMessagesHandler, MessageHandler, RawUpdateHandler
from pyrogram.raw.base import Update
from pyrogram.raw.functions.messages import DeleteHistory, ReadMentions
from pyrogram.raw.types import Channel, ChannelForbidden, UpdateChannel, User
from pyrogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Message,
    Story,
)
from pyrogram.utils import get_channel_id, timestamp_to_datetime

try:
    import uvloop
except ImportError:
    pass
else:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

ext = {
    "sleep_threshold": 900,
    "parse_mode": ParseMode.HTML,
    "link_preview_options": LinkPreviewOptions(is_disabled=True),
}

bot = Client(
    "bot",
    api_id=2040,
    api_hash="b18441a1ff607e10a989891a5462e627",
    bot_token=os.getenv("BOT_TOKEN"),
    workdir="sessions",
    no_updates=True,
    **ext,
)

apps = []
for i, ss in enumerate(os.getenv("SESSION_STRING").splitlines()):
    name = f"Client {i+1}"
    apps.append(Client(name=name, session_string=ss, **ext))

cmds = {}


async def event_log(client: Client, _: Update, __: User, chat: Channel) -> None:
    def fmt_date(timestamp: int) -> str:
        dt = timestamp_to_datetime(timestamp)
        return dt.strftime("%b %-d, %-H:%M %p")

    int32 = 2**31 - 1
    for _, event in chat.items():
        title, status, until = html.escape(event.title), None, None

        if isinstance(event, ChannelForbidden):
            status = "Channel Forbidden"

            if event.until_date and event.until_date < int32:
                until = fmt_date(event.until_date)

        elif isinstance(event, Channel):
            if event.admin_rights:
                status = "Privilege Updated"

            elif event.banned_rights:
                status = "Permission Updated"
                rights = event.banned_rights

                if rights.until_date and rights.until_date < int32:
                    until = fmt_date(rights.until_date)

            elif event.default_banned_rights:
                status = "Default Permission"
                rights = event.default_banned_rights

                if rights.until_date and rights.until_date < int32:
                    until = fmt_date(rights.until_date)

        if status:
            await bot.send_message(
                client.me.id,
                f"<pre language='{status}'>Title: {title}\nUntil: {until}</pre>",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "💬", url=f"tg://openmessage?chat_id={event.id}"
                            )
                        ]
                    ]
                ),
            )


cmds.update(
    {
        "Event Logger": RawUpdateHandler(
            event_log,
            filters.create(lambda _, __, event: isinstance(event, UpdateChannel)),
        )
    }
)


async def msg_log(client: Client, msg: Message) -> None:
    chat = msg.chat
    peer = await client.resolve_peer(chat.id)

    btn = None
    if chat.type == ChatType.PRIVATE:
        emo = "👤"
        if msg.from_user.is_contact:
            if not msg.media:
                return

            med = getattr(msg, msg.media.value)
            if hasattr(med, "ttl_seconds") and med.ttl_seconds:
                emo = "⌛️"
            else:
                return
        else:
            await client.invoke(DeleteHistory(peer=peer, max_id=msg.id, revoke=True))

        btn = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        emo, url=f"tg://openmessage?user_id={msg.from_user.id}"
                    )
                ]
            ]
        )

    else:
        await client.invoke(ReadMentions(peer=peer))

        url = f"https://t.me/c/{get_channel_id(chat.id)}/{msg.id}"
        if chat.is_forum:
            url = msg.link

        tmp = [InlineKeyboardButton("💬", url=url)]
        if msg.from_user:
            tmp.insert(
                0,
                InlineKeyboardButton(
                    "👤", url=f"tg://openmessage?user_id={msg.from_user.id}"
                ),
            )
        btn = InlineKeyboardMarkup([tmp])

    asyncio.create_task(send_log(client, msg, btn))


cmds.update(
    {
        "Mentioned Logger": MessageHandler(
            msg_log,
            (filters.private | filters.mentioned)
            & ~filters.me
            & ~filters.bot
            & ~filters.create(lambda _, __, msg: msg.from_user.is_support),
        )
    }
)


async def del_log(client: Client, messages: list[Message]) -> None:
    cache = client.message_cache.store

    for message in messages:
        if not message.chat:
            entry = next(
                (
                    (chat_id, msg_id)
                    for (chat_id, msg_id) in cache.keys()
                    if msg_id == message.id
                ),
                None,
            )
            if entry:
                msg = cache[entry]
                if msg.from_user.is_contact and not msg.outgoing:
                    btn = InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    "🗑",
                                    url=f"tg://openmessage?user_id={msg.from_user.id}",
                                )
                            ]
                        ]
                    )
                    asyncio.create_task(send_log(client, msg, btn))


cmds.update({"Deleted Logger": DeletedMessagesHandler(del_log)})


async def exec_cmd(client: Client, msg: Message) -> None:
    text = msg.text.split(maxsplit=1)
    if len(text) == 1:
        return

    cmd, code = text

    await msg.edit_text(f"<pre language=Running>{html.escape(code)}</pre>")
    start = time.perf_counter()

    async def aexec() -> None:
        pre, out = "", ""

        if cmd == "e":
            arg = {
                "asyncio": asyncio,
                "pyrogram": pyrogram,
                "enums": pyrogram.enums,
                "types": pyrogram.types,
                "utils": pyrogram.utils,
                "raw": pyrogram.raw,
                "c": client,
                "m": msg,
                "r": msg.reply_to_message,
                "u": (msg.reply_to_message or msg).from_user,
                "chat": msg.chat,
            }

            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                try:
                    res = await meval(code, globals(), **arg)
                    out = f.getvalue().strip() or str(res)
                except Exception as e:
                    pre, out = type(e).__name__, str(e)
                    if hasattr(e, "MESSAGE"):
                        out = str(e.MESSAGE).format(value=e.value)
        elif cmd == "sh":
            res = await asyncio.create_subprocess_shell(
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await res.communicate()
            out = (stdout + stderr).decode().strip()

        took = fmt_secs(time.perf_counter() - start)

        message = (
            f"<pre language=Input>{html.escape(code)}</pre>\n"
            f"<pre language='{pre}'>{html.escape(out)}</pre>\n"
            f"<pre language=Elapsed>{took}</pre>"
        )

        if len(message) > 1024:
            raw_url = await paste_rs(out)
            message = (
                f"<pre language=Input>{html.escape(code)}</pre>\n"
                f"<pre language='{pre}'>{html.escape(out[:512]) + '...'}</pre>\n"
                f"<blockquote><a href={raw_url}><b>More...</b></a></blockquote>\n"
                f"<pre language=Elapsed>{took}</pre>"
            )

        await msg.edit_text(message)

    task = asyncio.create_task(aexec())
    client.message_cache.store.update({msg.id: (task, code, start)})

    try:
        await task
    finally:
        client.message_cache.store.pop(msg.id, None)


cmds.update(
    {
        "Debug Commands": MessageHandler(
            exec_cmd, filters.me & filters.regex(r"^(e|sh)\s+.*")
        )
    }
)


async def task_cmd(client: Client, msg: Message) -> None:
    reply = msg.reply_to_message
    cache = client.message_cache.store.get(reply.id, None)

    if not (reply or cache):
        return

    task, code, start = cache
    task.cancel()

    took = fmt_secs(time.perf_counter() - start)
    await asyncio.gather(
        msg.edit_text(
            f"<pre language=Aborted>{html.escape(code)}</pre>\n"
            f"<pre language=Elapsed>{took}</pre>"
        ),
        reply.delete(),
    )


cmds.update(
    {"Abort Command": MessageHandler(task_cmd, filters.me & filters.regex(r"^x$"))}
)


async def del_cmd(_: Client, msg: Message) -> None:
    tasks, reply = [asyncio.create_task(msg.delete())], msg.reply_to_message
    if reply:
        tasks.append(asyncio.create_task(reply.delete()))

    await asyncio.gather(*tasks)


cmds.update(
    {"Delete Command": MessageHandler(del_cmd, filters.me & filters.regex(r"^d$"))}
)


async def purge_cmd(client: Client, msg: Message) -> None:
    chat = msg.chat
    text = msg.text

    async def search_msgs(min_id: int = 0, limit: int = 0, me: bool = False) -> list:
        ids = []
        async for message in client.search_messages(
            chat.id,
            limit=limit,
            from_user="me" if me else None,
            min_id=min_id,
            max_id=msg.id,
            message_thread_id=msg.message_thread_id if chat.is_forum else None,
        ):
            ids.append(message.id)
        return ids

    await msg.edit_text("<blockquote>Purging...</blockquote>")

    if text == "purge":
        _id = msg.reply_to_message_id
        if not _id:
            await msg.edit_text("<blockquote>Reply to Msg!</blockquote>")
            return await del_after(msg, 1)
        ids = await search_msgs(min_id=_id - 1)
    else:
        arg = text.split()
        if len(arg) < 2:
            ids = await search_msgs(me=True)
        else:
            _id = int(arg[1])
            ids = await search_msgs(limit=_id + 1, me=True)

    done = 0
    for batch in [ids[i : i + 100] for i in range(0, len(ids), 100)]:
        done += await client.delete_messages(chat.id, batch)

    message = f"{done} Message{'s' if done > 1 else ''} Deleted"
    await msg.edit_text(f"<blockquote>{message}</blockquote>")

    await del_after(msg, 2.5)


cmds.update(
    {
        "Purge Commands": MessageHandler(
            purge_cmd, filters.me & filters.regex(r"^(purge|purgeme(\s\d+)?)$")
        )
    }
)


async def send_log(
    client: Client,
    msg: Message,
    btn: InlineKeyboardMarkup = None,
    disable_notification: bool = False,
) -> Message:
    async def download(file_id: str) -> io.BytesIO:
        return await client.download_media(file_id, in_memory=True)

    app_id = client.me.id

    if msg.text:
        await bot.send_message(app_id, msg.text.html, reply_markup=btn)
        return

    med = getattr(msg, msg.media.value)

    if msg.sticker:
        await bot.send_sticker(app_id, med.file_id, reply_markup=btn)
        return

    if isinstance(med, Story):
        msg = await client.get_stories(med.chat.id, med.id)
        med = getattr(msg, msg.media.value)

    send_media = getattr(bot, f"send_{msg.media.value}")
    parameters = {
        "chat_id": app_id,
        "reply_markup": btn,
        **(
            {msg.media.value: await download(med.file_id)}
            if hasattr(med, "file_id")
            else {}
        ),
        **{
            key: getattr(med, key)
            for key in dir(med)
            if key in send_media.__annotations__.keys()
        },
        **({"caption": msg.caption.html} if msg.caption else {}),
    }

    if "thumb" in parameters and med.thumbs:
        parameters.update({"thumb": await download(med.thumbs[0].file_id)})

    for attr in ["view_once", "ttl_seconds"]:
        parameters.pop(attr, None)

    if disable_notification:
        parameters.update({"disable_notification": True})

    await send_media(**parameters)


async def paste_rs(content: str) -> str:
    async with aiohttp.ClientSession() as client:
        async with client.post("https://paste.rs", data=content) as resp:
            resp.raise_for_status()
            return await resp.text()


def fmt_secs(secs: int | float) -> str:
    if secs == 0:
        return "None"
    elif secs < 1e-3:
        return f"{secs * 1e6:.3f}".rstrip("0").rstrip(".") + " µs"
    elif secs < 1:
        return f"{secs * 1e3:.3f}".rstrip("0").rstrip(".") + " ms"
    return f"{secs:.3f}".rstrip("0").rstrip(".") + " s"


async def del_after(msg: Message, sleep: int = 0) -> None:
    await asyncio.sleep(sleep)
    await msg.delete()


if __name__ == "__main__":

    async def main() -> None:
        async def start(client: Client) -> None:
            await client.start()

            logger = logging.getLogger(client.name)
            for name, handler in cmds.items():
                client.add_handler(handler)
                logger.info(f"{name} Added")

        await bot.start()
        logging.info("Client Helper: Started")
        await asyncio.gather(*[asyncio.create_task(start(app)) for app in apps])

    aiorun.run(main(), loop=bot.loop)
