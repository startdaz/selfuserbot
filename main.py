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
from pyrogram.errors import MessageIdInvalid, PeerIdInvalid
from pyrogram.handlers import (
    DeletedMessagesHandler,
    EditedMessageHandler,
    MessageHandler,
    RawUpdateHandler,
)
from pyrogram.raw.base import Update
from pyrogram.raw.functions import Ping
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
        async for msg in client.get_chat_history(chat_id=user_id, limit=1):
            history = msg

    if not history:
        log = type(event.action).__name__
        btn = InlineKeyboardMarkup(
            [[InlineKeyboardButton(text="User", url=f"tg://user?id={user_id}")]]
        )

        await client.bot.send_message(
            chat_id=client.me.id,
            text=f"<pre language='User Action'>{log}</pre>",
            reply_markup=btn,
        )


flt_action = filters.create(
    lambda _, __, event: isinstance(event, UpdateUserTyping), "Flt Action"
)
cmds.update({"Action Log": RawUpdateHandler(callback=action_log, filters=flt_action)})


async def profile_log(client: Client, event: Update, _: User, __: Channel) -> None:
    user_id = event.user_id
    if user_id == client.me.id:
        return

    fmt = ""
    if event.last_name:
        fmt = f"\nLast Name : {html.escape(event.last_name)}"

    usernames = [obj.username for obj in event.usernames]
    if usernames:
        key = "Username  " if len(usernames) == 1 else "Usernames "
        fmt += f"\n{key}: {', '.join(usernames)}"

    btn = InlineKeyboardMarkup(
        [[InlineKeyboardButton(text="User", url=f"tg://user?id={user_id}")]]
    )
    log = f"First Name: {html.escape(event.first_name)}{fmt}"

    await client.bot.send_message(
        chat_id=client.me.id,
        text=f"<pre language='Profile Updated'>{log}</pre>",
        reply_markup=btn,
    )


flt_profile = filters.create(
    lambda _, __, event: isinstance(event, UpdateUserName), "Flt Profile"
)
cmds.update(
    {"Profile Log": RawUpdateHandler(callback=profile_log, filters=flt_profile)}
)


async def event_log(client: Client, _: Update, user: User, chat: Channel) -> None:
    def fmt_date(timestamp: int) -> str:
        dt = timestamp_to_datetime(timestamp)
        return dt.strftime("%b %-d, %-H:%M %p")

    int32 = 2**31 - 1

    if not user:
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
                log = title
                if until:
                    log = f"Title: {title}\nUntil: {until}"

                btn = InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                text="Open", url=f"tg://openmessage?chat_id={event.id}"
                            )
                        ]
                    ]
                )

                await client.bot.send_message(
                    chat_id=client.me.id,
                    text=f"<pre language='{status}'>{log}</pre>",
                    reply_markup=btn,
                )


flt_event = filters.create(
    lambda _, __, event: isinstance(event, UpdateChannel), "Flt Event"
)
cmds.update({"Event Log": RawUpdateHandler(callback=event_log, filters=flt_event)})


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
            if size_valid(msg) and not (msg.outgoing or msg.from_user.is_bot):
                btn = user_btn(text="Deleted", msg=msg)
                await send_log(msg=msg, btn=btn)


flt_deleted = filters.create(lambda _, __, msg: not msg.chat, "Flt Deleted")
cmds.update(
    {"Deleted Log": DeletedMessagesHandler(callback=deleted_log, filters=flt_deleted)}
)


async def edited_log(client: Client, msg: Message) -> None:
    if not size_valid(msg):
        return

    btn = chat_btn(text="Edited", msg=msg)
    await send_log(msg=msg, btn=btn)


cmds.update(
    {
        "Edited Log": EditedMessageHandler(
            callback=edited_log, filters=~filters.me & ~filters.bot & filters.mentioned
        )
    }
)


async def ping_cmd(client: Client, msg: Message) -> None:
    await msg.edit_text(text="<blockquote><b>...</b></blockquote>")

    ping = time.perf_counter()
    await client.invoke(Ping(ping_id=client.rnd_id()))
    pong = fmt_secs(secs=time.perf_counter() - ping)

    text = (
        "<a href='https://github.com/peerids/selfbot'><b>Selfbot</b></a>"
        f" - <b>Ping</b>\n<blockquote><b>{pong}</b></blockquote>"
    )
    await msg.edit_text(text=text)


cmds.update(
    {
        "Ping CMD": MessageHandler(
            callback=ping_cmd,
            filters=filters.me & ~filters.reply & filters.regex(r"^p$"),
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
            btn = user_btn(text="Limited", msg=msg)
    else:
        peer = await client.resolve_peer(peer_id=msg.chat.id)

        top = None
        if msg.chat.is_forum and msg.message_thread_id:
            top = msg.message_thread_id

        await client.invoke(ReadMentions(peer=peer, top_msg_id=top))

        btn = chat_btn(text="Message", msg=msg)

    if btn:
        await send_log(msg=msg, btn=btn)


cmds.update(
    {
        "Mentioned Log": MessageHandler(
            callback=mentioned_log,
            filters=~filters.me
            & ~filters.bot
            & ((filters.private & filters.media) | filters.mentioned),
        )
    }
)


async def debug_cmd(client: Client, msg: Message) -> None:
    cmd, code = msg.text.split(maxsplit=1)

    await msg.edit_text(text=f"<pre language=Running>{html.escape(code)}</pre>")
    start = time.perf_counter()

    async def aexec() -> None:
        async def play(
            media_path: str,
            chat_id: int | str = msg.chat.id,
            video: bool = False,
            **kwargs,
        ) -> None:
            await client.call.play(
                chat_id=chat_id,
                stream=MediaStream(
                    media_path=media_path,
                    video_flags=None if video else MediaStream.Flags.IGNORE,
                    ytdlp_parameters="--cookies cookies.txt",
                    **kwargs,
                ),
            )

        async def paste(content: str) -> str:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"https://paste.rs", data=content) as post:
                    post.raise_for_status()
                    return await post.text()

        pre, out = "python", None

        if cmd == "e":
            kwargs = {
                "asyncio": asyncio,
                "pyrogram": pyrogram,
                "enums": pyrogram.enums,
                "errors": pyrogram.errors,
                "types": pyrogram.types,
                "utils": pyrogram.utils,
                "raw": pyrogram.raw,
                "cache": client.message_cache.store,
                "c": client,
                "m": msg,
                "r": msg.reply_to_message,
                "u": (msg.reply_to_message or msg).from_user,
                "chat": msg.chat,
                "call": client.call,
                "play": play,
            }

            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                try:
                    res = await meval(code=code, globs=globals(), **kwargs)
                    out = f.getvalue().strip() or str(res)
                except Exception as e:
                    pre, out = type(e).__name__, str(e)
                    if hasattr(e, "MESSAGE"):
                        out = str(e.MESSAGE).format(value=e.value)
        elif cmd == "sh":
            pre = "bash"
            res = await asyncio.create_subprocess_shell(
                cmd=code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await res.communicate()
            out = (stdout + stderr).decode().strip()

        took = fmt_secs(secs=time.perf_counter() - start)

        text = f"<pre language='{pre}'>{html.escape(out)}</pre>"
        if len(text) > 1024:
            more = None
            if len(code) > 1024:
                more = await paste(content=f"In:\n{code}\n\nOut:\n{out}")
            else:
                more = await paste(content=out)

            text = (
                f"<pre language='{pre}'>{html.escape(out[:512])}...</pre>"
                f"<blockquote><a href={more}><b>More...</b></a></blockquote>"
            )

        fmt_in = f"{code[:512]}..." if len(code) > 1024 else code
        result = (
            f"<pre language=Input>{html.escape(fmt_in)}</pre>\n"
            f"{text}\n<pre language=Elapsed>{took}</pre>"
        )
        await msg.edit_text(text=result)

    name = f"{client.me.id}_{msg.chat.id}_{msg.id}"
    task = asyncio.create_task(coro=aexec(), name=name)
    client.message_cache.store.update({name: task})

    try:
        await task
    except asyncio.CancelledError:
        took = fmt_secs(secs=time.perf_counter() - start)

        with contextlib.suppress(MessageIdInvalid):
            text = (
                f"<pre language=Aborted>{html.escape(code)}</pre>\n"
                f"<pre language=Elapsed>{took}</pre>"
            )
            await msg.edit_text(text=text)
    finally:
        client.message_cache.store.pop(name, None)


cmds.update(
    {
        "Debug CMD": MessageHandler(
            callback=debug_cmd, filters=filters.me & filters.regex(r"^(e|sh)\s+.*")
        )
    }
)


async def abort_cmd(client: Client, msg: Message) -> None:
    await msg.edit_text(text="<blockquote><b>Aborting...</b></blockquote>")

    tasks = []
    if msg.reply_to_message_id:
        task = client.message_cache.store.get(
            f"{client.me.id}_{msg.chat.id}_{msg.reply_to_message_id}"
        )
        if task:
            tasks = [task]

    else:
        for task in asyncio.all_tasks():
            if task.get_name().startswith(f"{client.me.id}_"):
                tasks.append(task)

    if not tasks:
        await msg.edit_text(text="<blockquote><b>None</b></blockquote>")
        return

    for task in tasks:
        task.cancel()

    text = f"{len(tasks)} Task{'s' if len(tasks) > 1 else ''} Aborted"
    await msg.edit_text(text=f"<blockquote><b>{text}</b></blockquote>")


cmds.update(
    {
        "Abort CMD": MessageHandler(
            callback=abort_cmd, filters=filters.me & filters.regex(r"^x$")
        )
    }
)


async def delete_cmd(_: Client, msg: Message) -> None:
    await asyncio.gather(
        msg.delete(revoke=True), msg.reply_to_message.delete(revoke=True)
    )


cmds.update(
    {
        "Delete CMD": MessageHandler(
            callback=delete_cmd,
            filters=filters.me & filters.reply & filters.regex(r"^d$"),
        )
    }
)


async def purge_cmd(client: Client, msg: Message) -> None:
    chat, text = msg.chat, msg.text

    async def search_msgs(min_id: int = 0, limit: int = 0, me: bool = False) -> list:
        ids = []
        async for message in client.search_messages(
            chat_id=chat.id,
            limit=limit,
            from_user="me" if me else None,
            min_id=min_id,
            max_id=msg.id,
            message_thread_id=msg.message_thread_id if chat.is_forum else None,
        ):
            ids.append(message.id)
        return ids

    async def del_after(sleep: int = 0) -> None:
        await asyncio.sleep(delay=sleep)
        await msg.delete(revoke=True)

    await msg.edit_text(text="<blockquote><b>Purging...</b></blockquote>")

    if text == "purge":
        ids = await search_msgs(min_id=msg.reply_to_message_id - 1)
    else:
        args = text.split()
        if len(args) < 2:
            ids = await search_msgs(me=True)
        else:
            _id = int(args[1])
            ids = await search_msgs(limit=_id + 1, me=True)

    done = 0
    for batch in [ids[i : i + 100] for i in range(0, len(ids), 100)]:
        done += await client.delete_messages(
            chat_id=chat.id, message_ids=batch, revoke=True
        )

    result = f"{done} Message{'s' if done > 1 else ''} Deleted"
    await msg.edit_text(text=f"<blockquote><b>{result}</b></blockquote>")

    await del_after(sleep=2.5)


cmds.update(
    {
        "Purge CMD": MessageHandler(
            callback=purge_cmd,
            filters=filters.me & (filters.regex(r"^purge$") & filters.reply)
            | (filters.regex(r"^purgeme(\s\d+)?$") & ~filters.reply),
        )
    }
)


def user_btn(text: str, msg: Message) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    text=text,
                    url=f"tg://openmessage?user_id={msg.from_user.id}&message_id={msg.id}",
                )
            ]
        ]
    )


def chat_btn(text: str, msg: Message) -> InlineKeyboardMarkup:
    url = f"https://t.me/c/{get_channel_id(msg.chat.id)}/{msg.id}"
    if msg.chat.is_forum:
        url = msg.link

    tmp = [[InlineKeyboardButton(text=text, url=url)]]
    if msg.from_user:
        tmp.insert(
            0,
            [InlineKeyboardButton(text="User", url=f"tg://user?id={msg.from_user.id}")],
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
        return await app.download_media(message=file_id, in_memory=True)

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
            chat_id=app_id, text=msg.text.html, reply_markup=btn, reply_parameters=rep
        )
        caches.update({key_id: sent.id})
        return

    obj = getattr(msg, msg.media.value)

    if msg.sticker:
        await bot.send_sticker(
            chat_id=app_id, sticker=obj.file_id, reply_markup=btn, reply_parameters=rep
        )
        return

    if isinstance(obj, Story):
        msg = await app.get_stories(story_sender_chat_id=obj.chat.id, story_ids=obj.id)
        obj = getattr(msg, msg.media.value)

    send_media = getattr(bot, f"send_{msg.media.value}")
    parameters = {
        "chat_id": app_id,
        "reply_markup": btn,
        "reply_parameters": rep,
        **(
            {msg.media.value: await download(file_id=obj.file_id)}
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
        parameters.update({"thumb": await download(file_id=obj.thumbs[0].file_id)})

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
            subprocess.run([sys.executable, "-m", "pip", "install", "-U", "pip"])
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-U", "-r", "requirements.txt"]
            )

            os.system("clear")

        async def start(client: Client) -> None:
            await client.start()

            logger = logging.getLogger(str(client.me.id))
            for group, (name, handler) in enumerate(cmds.items(), start=1):
                client.add_handler(handler=handler, group=group)
                logger.info(f"{name} Added")

            setattr(client, "call", PyTgCalls(client))
            await client.call.start()

            limit = await client.get_dialogs_count()
            async for _ in client.get_dialogs(limit=limit):
                pass

            await client.storage.save()

        sys_update()
        logging.info("System Up to Date")

        await bot.start()
        logging.info("Client Helper Started")

        logging.info(f"Starting {len(apps)} App Clients")
        await asyncio.gather(*[asyncio.create_task(coro=start(client=i)) for i in apps])

    aiorun.logger.disabled = True
    aiorun.run(main(), loop=bot.loop)
