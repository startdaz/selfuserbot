import asyncio
import contextlib
import html
import io
import logging
import os
import subprocess
import sys
import time

import aiohttp
import aiorun
import pyrogram
from meval import meval
from pyrogram import Client, filters
from pyrogram.enums import ChatType, ParseMode
from pyrogram.errors import PeerIdInvalid
from pyrogram.handlers import (
    DeletedMessagesHandler,
    EditedMessageHandler,
    MessageHandler,
    RawUpdateHandler,
)
from pyrogram.raw.base import Update
from pyrogram.raw.functions.messages import ReadMentions
from pyrogram.raw.types import (
    Channel,
    ChannelForbidden,
    UpdateChannel,
    UpdateUserName,
    UpdateUserTyping,
    User,
)
from pyrogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Message,
    ReplyParameters,
    Story,
)
from pyrogram.utils import get_channel_id, timestamp_to_datetime
from pytgcalls import PyTgCalls
from pytgcalls.types import MediaStream

try:
    import uvloop
except ImportError:
    pass
else:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

cmds = {}


async def action_log(client: Client, event: Update, _: User, __: Channel) -> None:
    user_id = event.user_id

    history = None
    with contextlib.suppress(PeerIdInvalid):
        async for msg in client.get_chat_history(user_id, limit=1):
            history = msg

    if not history:
        act = type(event.action).__name__
        btn = InlineKeyboardMarkup(
            [[InlineKeyboardButton("User", url=f"tg://user?id={user_id}")]]
        )

        await client.bot.send_message(
            client.me.id, f"<pre language='User Action'>{act}</pre>", reply_markup=btn
        )


flt_action = filters.create(
    lambda _, __, event: isinstance(event, UpdateUserTyping), "Flt Action"
)
cmds.update({"Action Log": RawUpdateHandler(action_log, flt_action)})


async def profile_log(client: Client, event: Update, _: User, __: Channel) -> None:
    await client.bot.send_message(
        client.me.id,
        (
            "<pre language='Profile Updated'>"
            f"First Name: {html.escape(event.first_name)}\n"
            f"Last Name : {html.escape(event.last_name)}\n"
            f"Username  : {event.usernames}</pre>"
        ),
        reply_markup=InlineKeyboardMarkup(
            [[InlineKeyboardButton("User", url=f"tg://user?id={event.user_id}")]]
        ),
    )


flt_profile = filters.create(
    lambda _, __, event: isinstance(event, UpdateUserName), "Flt Profile"
)
cmds.update({"Profile Log": RawUpdateHandler(profile_log, flt_profile)})


async def event_log(client: Client, _: Update, __: User, chat: Channel) -> None:
    def fmt_date(timestamp: int) -> str:
        dt = timestamp_to_datetime(timestamp)
        return dt.strftime("%b %-d, %-H:%M %p")

    int32 = 2**31 - 1

    for _, event in chat.items():
        title = html.escape(event.title)
        status, until = None, None

        if isinstance(event, ChannelForbidden):
            status = "Channel Forbidden"
            if event.until_date and event.until_date < int32:
                until = fmt_date(event.until_date)

        elif isinstance(event, Channel):
            if event.admin_rights:
                status = "Privilege Updated"

            elif event.banned_rights:
                status = "Permission Updated"
                if (
                    event.banned_rights.until_date
                    and event.banned_rights.until_date < int32
                ):
                    until = fmt_date(event.banned_rights.until_date)

            elif event.default_banned_rights:
                status = "Default Permission"
                if (
                    event.default_banned_rights.until_date
                    and event.default_banned_rights.until_date < int32
                ):
                    until = fmt_date(event.default_banned_rights.until_date)

        if status:
            await client.bot.send_message(
                client.me.id,
                f"<pre language='{status}'>Title: {title}\nUntil: {until}</pre>",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Chat", url=f"tg://openmessage?chat_id={event.id}"
                            )
                        ]
                    ]
                ),
            )


flt_event = filters.create(
    lambda _, __, event: isinstance(event, UpdateChannel), "Flt Event"
)
cmds.update({"Event Log": RawUpdateHandler(event_log, flt_event)})


async def deleted_log(client: Client, messages: list[Message]) -> None:
    cache = client.message_cache.store

    for message in messages:
        entry = next(
            (
                (_, message_id)
                for (_, message_id) in cache.keys()
                if message_id == message.id
            ),
            None,
        )

        if entry:
            msg = cache[entry]

            if not (msg.outgoing or msg.from_user.is_bot):
                if size_valid(msg):
                    btn = user_btn("Deleted", msg)
                    asyncio.create_task(send_log(msg, btn))


flt_deleted = filters.create(lambda _, __, msg: not msg.chat, "Flt Deleted")
cmds.update({"Deleted Log": DeletedMessagesHandler(deleted_log, flt_deleted)})


async def edited_log(client: Client, msg: Message) -> None:
    if not size_valid(msg):
        return

    btn = chat_btn("Edited", msg)
    asyncio.create_task(send_log(msg, btn))


cmds.update(
    {
        "Edited Log": EditedMessageHandler(
            edited_log, ~filters.me & ~filters.bot & filters.mentioned
        )
    }
)


async def mentioned_log(client: Client, msg: Message) -> None:
    if not size_valid(msg):
        return

    btn = None

    if msg.chat.type == ChatType.PRIVATE:
        obj = getattr(msg, msg.media.value)
        if hasattr(obj, "ttl_seconds") and obj.ttl_seconds:
            btn = user_btn("Limited", msg)
    else:
        peer = await client.resolve_peer(msg.chat.id)

        top = None
        if msg.chat.is_forum and msg.message_thread_id:
            top = msg.message_thread_id

        await client.invoke(ReadMentions(peer=peer, top_msg_id=top))

        btn = chat_btn("Message", msg)

    if btn:
        asyncio.create_task(send_log(msg, btn))


cmds.update(
    {
        "Mentioned Log": MessageHandler(
            mentioned_log,
            ~filters.me
            & ~filters.bot
            & ((filters.private & filters.media) | filters.mentioned),
        )
    }
)


async def debug_cmd(client: Client, msg: Message) -> None:
    cmd, code = msg.text.split(maxsplit=1)

    await msg.edit_text(f"<pre language=Running>{html.escape(code)}</pre>")
    start = time.perf_counter()

    async def aexec() -> None:
        async def play(url: str, video: bool = False, **kwargs) -> None:
            await client.call.play(
                msg.chat.id,
                MediaStream(
                    url,
                    video_flags=None if video else MediaStream.Flags.IGNORE,
                    ytdlp_parameters="--cookies cookies.txt",
                    **kwargs,
                ),
            )

        async def paste(content: str) -> tuple:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"https://paste.rs", data=content) as post:
                    post.raise_for_status()
                    return await post.text()

        def truncate(text: str, limit: int) -> str:
            if len(text) > limit:
                if limit > 3:
                    return text[: limit - 3] + "..."
                return text[:limit]
            return text

        pre, out = None, None

        if cmd == "e":
            pre = "python"
            arg = {
                "asyncio": asyncio,
                "pyrogram": pyrogram,
                "enums": pyrogram.enums,
                "types": pyrogram.types,
                "utils": pyrogram.utils,
                "raw": pyrogram.raw,
                "cache": client.message_cache.store,
                "c": client,
                "m": msg,
                "r": msg.reply_to_message,
                "u": (msg.reply_to_message or msg).from_user,
                "chat": msg.chat,
                "play": play,
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
            pre = "bash"
            res = await asyncio.create_subprocess_shell(
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await res.communicate()
            out = (stdout + stderr).decode().strip()

        took = fmt_secs(time.perf_counter() - start)

        text = f"<pre language='{pre}'>{html.escape(out)}</pre>"
        if len(text) > 1024:
            more = None
            if len(code) > 1024:
                more = await paste(f"In:\n{code}\n\nOut:\n{out}")
            else:
                more = await paste(out)

            text = (
                f"<pre language='{pre}'>{html.escape(out[:512])}...</pre>"
                f"<blockquote><a href={more}><b>More...</b></a></blockquote>"
            )

        _in = f"{code[:512]}..." if len(code) > 1024 else code
        await msg.edit_text(
            f"<pre language=Input>{html.escape(_in)}</pre>\n"
            f"{text}\n<pre language=Elapsed>{took}</pre>"
        )

    task = asyncio.create_task(aexec())
    client.message_cache.store.update({msg.id: (code, start, task)})

    try:
        await task
    finally:
        client.message_cache.store.pop(msg.id, None)


cmds.update(
    {
        "Debug CMD": MessageHandler(
            debug_cmd, filters.me & filters.regex(r"^(e|sh)\s+.*")
        )
    }
)


async def abort_cmd(_: Client, msg: Message) -> None:
    reply = msg.reply_to_message

    cache = msg._client.message_cache.store.get(reply.id, None)
    if not cache:
        return

    code, start, task = cache
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
    {
        "Abort CMD": MessageHandler(
            abort_cmd, filters.me & filters.reply & filters.regex(r"^x$")
        )
    }
)


async def delete_cmd(_: Client, msg: Message) -> None:
    await asyncio.gather(msg.delete(), msg.reply_to_message.delete())


cmds.update(
    {
        "Delete CMD": MessageHandler(
            delete_cmd, filters.me & filters.reply & filters.regex(r"^d$")
        )
    }
)


async def purge_cmd(client: Client, msg: Message) -> None:
    chat, text = msg.chat, msg.text

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

    async def del_after(sleep: int = 0) -> None:
        await asyncio.sleep(sleep)
        await msg.delete()

    await msg.edit_text("<blockquote>Purging...</blockquote>")

    if text == "purge":
        ids = await search_msgs(min_id=msg.reply_to_message_id - 1)
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

    await del_after(2.5)


cmds.update(
    {
        "Purge CMD": MessageHandler(
            purge_cmd,
            filters.me
            & (
                (filters.regex(r"^purge$") & filters.reply)
                | (filters.regex(r"^purgeme(\s\d+)?$") & ~filters.reply)
            ),
        )
    }
)


def user_btn(title: str, msg: Message) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    title,
                    url=f"tg://openmessage?user_id={msg.from_user.id}&message_id={msg.id}",
                )
            ]
        ]
    )


def chat_btn(title: str, msg: Message) -> InlineKeyboardMarkup:
    url = f"https://t.me/c/{get_channel_id(msg.chat.id)}/{msg.id}"
    if msg.chat.is_forum:
        url = msg.link

    tmp = [[InlineKeyboardButton(title, url=url)]]
    if msg.from_user:
        tmp.insert(
            0,
            [InlineKeyboardButton("User", url=f"tg://user?id={msg.from_user.id}")],
        )

    return InlineKeyboardMarkup(tmp)


def size_valid(msg: Message) -> bool:
    if msg.media:
        obj = getattr(msg, msg.media.value)
        if hasattr(obj, "file_size") and obj.file_size:
            if obj.file_size > 64 * (1024**2):
                return False
    return True


async def send_log(msg: Message, btn: InlineKeyboardMarkup) -> None:
    app, bot = msg._client, msg._client.bot

    async def download(file_id: str) -> io.BytesIO:
        return await app.download_media(file_id, in_memory=True)

    if msg.service:
        return None

    app_id = app.me.id
    caches = bot.message_cache.store

    rep = None

    key_id = (app_id, msg.chat.id, msg.id)
    if key_id in caches.keys():
        old = caches.get(key_id)
        rep = ReplyParameters(message_id=old)
        btn = None

    if msg.text:
        sent = await bot.send_message(
            app_id, msg.text.html, reply_markup=btn, reply_parameters=rep
        )
        caches.update({key_id: sent.id})
        return

    obj = getattr(msg, msg.media.value)

    if msg.sticker:
        await bot.send_sticker(
            app_id, obj.file_id, reply_markup=btn, reply_parameters=rep
        )
        return

    if isinstance(obj, Story):
        msg = await app.get_stories(obj.chat.id, obj.id)
        obj = getattr(msg, msg.media.value)

    send_media = getattr(bot, f"send_{msg.media.value}")
    parameters = {
        "chat_id": app_id,
        "reply_markup": btn,
        "reply_parameters": rep,
        **(
            {msg.media.value: await download(obj.file_id)}
            if hasattr(obj, "file_id")
            else {}
        ),
        **{
            key: getattr(obj, key)
            for key in dir(obj)
            if key in send_media.__annotations__.keys()
        },
        **({"caption": msg.caption.html} if msg.caption else {}),
    }

    if "thumb" in parameters and obj.thumbs:
        parameters.update({"thumb": await download(obj.thumbs[0].file_id)})

    for attr in ["view_once", "ttl_seconds"]:
        parameters.pop(attr, None)

    sent = await send_media(**parameters)
    caches.update({key_id: sent.id})


def fmt_secs(secs: int | float) -> str:
    if secs == 0:
        return "None"
    elif secs < 1e-3:
        return f"{secs * 1e6:.3f}".rstrip("0").rstrip(".") + " µs"
    elif secs < 1:
        return f"{secs * 1e3:.3f}".rstrip("0").rstrip(".") + " ms"
    return f"{secs:.3f}".rstrip("0").rstrip(".") + " s"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s - %(name)s: %(message)s"
    )
    logging.getLogger("pyrogram").setLevel(logging.ERROR)

    ext = {
        "parse_mode": ParseMode.HTML,
        "sleep_threshold": 900,
        "max_message_cache_size": 2147483647,
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
    for i, ss in enumerate(os.getenv("SESSION_STRING").split()):
        app = Client(name=str(i), session_string=ss, **ext)
        setattr(app, "bot", bot)
        apps.append(app)

    async def main() -> None:
        def sys_update() -> None:
            subprocess.run(
                ["git", "fetch"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            subprocess.run(
                ["git", "reset", "--hard", "origin/master"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-U", "pip"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-U",
                    "-r",
                    "requirements.txt",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        async def start(client: Client) -> None:
            await client.start()

            logger = logging.getLogger(str(client.me.id))
            for name, handler in cmds.items():
                client.add_handler(handler)
                logger.info(f"{name} Added")

            setattr(client, "call", PyTgCalls(client))
            await client.call.start()

            limit = await client.get_dialogs_count()
            async for _ in client.get_dialogs(limit=limit):
                pass

            await client.storage.save()

        sys_update()

        await bot.start()
        logging.info("Client Helper Started")

        logging.info(f"Starting {len(apps)} App Clients")
        await asyncio.gather(*[asyncio.create_task(start(i)) for i in apps])

    aiorun.logger.disabled = True
    aiorun.run(main(), loop=bot.loop)
