import asyncio
import collections
import contextlib
import html
import io
import logging
import os
import time
import sys

import aiohttp
import aiorun
import pyrogram
from meval import meval
from pyrogram import Client, filters
from pyrogram.enums import ChatType, ParseMode
from pyrogram.errors import MessageIdInvalid, PeerIdInvalid, FloodWait
from pyrogram.handlers import (
    DeletedMessagesHandler,
    EditedMessageHandler,
    MessageHandler,
    RawUpdateHandler,
)
from pyrogram.raw.base import Update
from pyrogram.raw.functions.messages import DeleteHistory
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
    InputMediaPhoto,
    InputMediaVideo,
)
from pyrogram.utils import get_channel_id, timestamp_to_datetime

try:
    import uvloop
except ImportError:
    pass
else:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

bot = Client(
    "bot",
    api_id=2040,
    api_hash="b18441a1ff607e10a989891a5462e627",
    bot_token=os.getenv("BOT_TOKEN"),
    no_updates=True,
    parse_mode=ParseMode.HTML,
    sleep_threshold=86400,
    max_message_cache_size=2**31 - 1,
    link_preview_options=LinkPreviewOptions(is_disabled=True),
)

cmds = {}

async def action_log(client: Client, update: Update, _: User, __: Channel) -> None:
    user_id = update.user_id

    history = None
    with contextlib.suppress(PeerIdInvalid):
        async for msg in client.get_chat_history(chat_id=user_id, limit=1):
            history = msg

    if not history:
        log = type(update.action).__name__
        btn = InlineKeyboardMarkup(
            [[InlineKeyboardButton(text="User", url=f"tg://user?id={user_id}")]]
        )

        await bot.send_message(
            chat_id=client.me.id,
            text=f"<pre language='User Action'>{log}</pre>",
            reply_markup=btn,
        )


cmds.update(
    {
        "Action Log": RawUpdateHandler(
            callback=action_log,
            filters=filters.create(
                lambda _, __, update: isinstance(update, UpdateUserTyping),
                name="Filter Action Log",
            ),
        )
    }
)


async def profile_log(client: Client, update: Update, _: User, __: Channel) -> None:
    user_id = update.user_id
    if user_id == client.me.id:
        return

    fmt = ""
    if update.last_name:
        fmt = f"\nLast Name : {html.escape(update.last_name)}"

    usernames = [obj.username for obj in update.usernames]
    if usernames:
        key = "Username  " if len(usernames) == 1 else "Usernames "
        fmt += f"\n{key}: {', '.join(usernames)}"

    btn = InlineKeyboardMarkup(
        [[InlineKeyboardButton(text="User", url=f"tg://user?id={user_id}")]]
    )
    log = f"First Name: {html.escape(update.first_name)}{fmt}"

    await bot.send_message(
        chat_id=client.me.id,
        text=f"<pre language='Profile Updated'>{log}</pre>",
        reply_markup=btn,
    )


cmds.update(
    {
        "Profile Log": RawUpdateHandler(
            callback=profile_log,
            filters=filters.create(
                lambda _, __, update: isinstance(update, UpdateUserName),
                name="Filter Profile Log",
            ),
        )
    }
)


async def event_log(client: Client, _: Update, user: User, channel: Channel) -> None:
    def fmt_date(timestamp: int) -> str:
        dt = timestamp_to_datetime(timestamp)
        return dt.strftime("%b %-d, %-I:%M %p")

    if not user:
        for _, event in channel.items():
            title = html.escape(event.title)
            status, until = None, None

            if isinstance(event, ChannelForbidden):
                status = "Channel Forbidden"
                if (
                    event.until_date
                    and event.until_date < client.max_message_cache_size
                ):
                    until = fmt_date(event.until_date)
            elif isinstance(event, Channel):
                if event.admin_rights:
                    status = "Privilege Updated"
                elif event.banned_rights:
                    status = "Permission Updated"
                    if (
                        event.banned_rights.until_date
                        and event.banned_rights.until_date
                        < client.max_message_cache_size
                    ):
                        until = fmt_date(event.banned_rights.until_date)
                elif event.default_banned_rights:
                    status = "Default Permission"
                    if (
                        event.default_banned_rights.until_date
                        and event.default_banned_rights.until_date
                        < client.max_message_cache_size
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

                await bot.send_message(
                    chat_id=client.me.id,
                    text=f"<pre language='{status}'>{log}</pre>",
                    reply_markup=btn,
                )


cmds.update(
    {
        "Event Log": RawUpdateHandler(
            callback=event_log,
            filters=filters.create(
                lambda _, __, update: isinstance(update, UpdateChannel),
                name="Filter Event Log",
            ),
        )
    }
)


async def deleted_log(client: Client, messages: list[Message]) -> None:
    cache = client.message_cache.store

    for message in messages:
        msg = None

        if message.chat:
            k = (message.id, message.chat.id)
            if k in cache:
                msg = cache.get(k)
        else:
            k = next(
                (
                    (message_id, _)
                    for (message_id, _) in cache.keys()
                    if message_id == message.id
                ),
                None,
            )
            if k:
                msg = cache.get(k)

        if msg:
            bot.message_cache.store.pop((client.me.id, msg.chat.id, msg.id), None)

            btn = log_btn(msg=msg, text="Deleted")
            await send_log(msg=msg, btn=btn)

            cache.pop((msg.id, msg.chat.id), None)


cmds.update({"Deleted Log": DeletedMessagesHandler(callback=deleted_log)})


async def edited_log(client: Client, msg: Message) -> None:
    k, v, cache = (msg.id, msg.chat.id), msg, client.message_cache.store
    if k in cache:
        msg = cache.get((msg.id, msg.chat.id))

    cache.update({k: v})

    btn = log_btn(msg=msg, text="Edited")
    await send_log(msg=msg, btn=btn)

    cache.pop((msg.chat.id, msg.id), None)


cmds.update(
    {
        "Edited Log": EditedMessageHandler(
            callback=edited_log,
            filters=~filters.me & ~filters.bot & (filters.mentioned | filters.private),
        )
    }
)


async def incoming_log(client: Client, msg: Message) -> None:
    cache = client.message_cache.store
    cache.update({(msg.id, msg.chat.id): msg})
    
    if msg.chat.type == ChatType.PRIVATE:
        btn_text = "Message from Contact" if msg.from_user.is_contact else "From Non-Contact"
    else:
        btn_text = "Incoming Message"

    btn = log_btn(msg=msg, text=btn_text)
    
    if msg.media:
        obj = getattr(msg, msg.media.value)
        if hasattr(obj, "ttl_seconds") and obj.ttl_seconds:
            btn = log_btn(msg=msg, text="Self-Destruct Media")

    await send_log(msg=msg, btn=btn)
    cache.pop((msg.chat.id, msg.id), None)

cmds.update(
    {
        "Incoming Log": MessageHandler(
            callback=incoming_log,
            filters=~filters.me & ~filters.bot & (filters.mentioned | filters.private),
        )
    }
)



#async def add_contact(client: Client, msg: Message) -> None:
#    user = await client.get_users(msg.chat.id)
#    if not (user.is_contact or user.is_support):
#        await client.add_contact(
#            user_id=user.id,
#            first_name=user.first_name,
#            last_name=user.last_name or "",
#        )
#
#    await msg.continue_propagation()


#cmds.update(
#    {
#        "Add Contact": MessageHandler(
#            callback=add_contact, filters=filters.me & filters.private & ~filters.bot
#        )
#    }
#)


async def debug_cmd(client: Client, msg: Message) -> None:
    cmd, code = msg.text.split(maxsplit=1)

    await msg.edit_text(text=f"<pre language=Running>{html.escape(code)}</pre>")
    start = time.perf_counter()

    async def aexec() -> None:
        async def paste_rs(content: str) -> str:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"https://paste.rs", data=content) as post:
                    post.raise_for_status()
                    return await post.text()

        pre, out = "", None

        if cmd == "e":
            kwargs = {
                "os": os,
                "sys": sys,
                "asyncio": asyncio,
                "pyrogram": pyrogram,
                "bot": bot,
                "c": client,
                "m": msg,
                "r": msg.reply_to_message,
                "u": (msg.reply_to_message or msg).from_user,
                "InputMediaPhoto": InputMediaPhoto,
                "InputMediaVideo": InputMediaVideo,
                "FloodWait": FloodWait,
            }

            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                client.max_concurrent_transmissions += 1
                try:
                    res = await meval(code=code, globs=globals(), **kwargs)
                except Exception as e:
                    pre, out = type(e).__name__, str(e)
                    if hasattr(e, "MESSAGE"):
                        out = str(e.MESSAGE).format(value=e.value)
                else:
                    out = f.getvalue().strip() or str(res)
                finally:
                    client.max_concurrent_transmissions -= 1
        elif cmd == "sh":
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
                more = await paste_rs(content=f"In:\n{code}\n\nOut:\n{out}")
            else:
                more = await paste_rs(content=out)

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

    try:
        await asyncio.wait_for(fut=task, timeout=900)
    except (asyncio.CancelledError, asyncio.TimeoutError) as e:
        if isinstance(e, asyncio.TimeoutError):
            task.cancel()

        took = fmt_secs(secs=time.perf_counter() - start)
        text = (
            f"<pre language={type(e).__name__}>{html.escape(code)}</pre>\n"
            f"<pre language=Elapsed>{took}</pre>"
        )
        with contextlib.suppress(MessageIdInvalid):
            await msg.edit_text(text=text)
    finally:
        del task


cmds.update(
    {
        "Debug CMD": MessageHandler(
            callback=debug_cmd,
            filters=filters.me & filters.regex(r"^(e|sh)\s+.*"),
        )
    }
)


async def tasks_cmd(client: Client, msg: Message) -> None:
    names = [
        task.get_name()
        for task in asyncio.all_tasks()
        if task.get_name().startswith(f"{client.me.id}_")
    ]

    if not names:
        await msg.edit_text("<blockquote><b>None</b></blockquote>")
        return await del_after(msg=msg, delay=2.5)

    text = "<blockquote><b>List of Tasks</b></blockquote>\n"

    urls = []
    for i, name in enumerate(names, start=1):
        _, chat_id, msg_id = name.split("_")

        if chat_id.startswith("-100"):
            url = f"t.me/c/{get_channel_id(int(chat_id))}/{msg_id}"
        else:
            url = f"tg://openmessage?user_id={chat_id}&message_id={msg_id}"

        urls.append(f" <a href={url}><b>Task - {i}</b></a>")

    text += "\n".join(urls)
    await msg.edit_text(text=text)


cmds.update(
    {
        "Tasks CMD": MessageHandler(
            callback=tasks_cmd,
            filters=filters.me & ~filters.reply & filters.regex(r"^tasks$"),
        )
    }
)


async def abort_cmd(client: Client, msg: Message) -> None:
    await msg.edit_text(text="<blockquote><b>Aborting...</b></blockquote>")

    tasks = []
    for task in asyncio.all_tasks():
        if msg.reply_to_message_id:
            if (
                task.get_name()
                == f"{client.me.id}_{msg.chat.id}_{msg.reply_to_message_id}"
            ):
                tasks = [task]
                break
        else:
            if task.get_name().startswith(f"{client.me.id}_"):
                tasks.append(task)

    if not tasks:
        await msg.edit_text(text="<blockquote><b>None</b></blockquote>")
        return await del_after(msg=msg, delay=2.5)

    for task in tasks:
        task.cancel()

    text = f"{len(tasks)} Task{'s' if len(tasks) > 1 else ''} Aborted"
    await msg.edit_text(text=f"<blockquote><b>{text}</b></blockquote>")

    await del_after(msg=msg, delay=2.5)


cmds.update(
    {
        "Abort CMD": MessageHandler(
            callback=abort_cmd,
            filters=filters.me & filters.regex(r"^x$"),
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

    await del_after(msg=msg, delay=2.5)


cmds.update(
    {
        "Purge CMD": MessageHandler(
            callback=purge_cmd,
            filters=filters.me
            & (
                (filters.regex(r"^purge$") & filters.reply)
                | (filters.regex(r"^purgeme(\s\d+)?$") & ~filters.reply)
            ),
        )
    }
)


def log_btn(msg: Message, text: str = "Message") -> InlineKeyboardMarkup:
    if msg.chat.type == ChatType.PRIVATE:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text=text,
                        url=f"tg://openmessage?user_id={msg.chat.id}&message_id={msg.id}",
                    )
                ]
            ]
        )
    else:
        channel_id = get_channel_id(msg.chat.id)

        url = f"t.me/c/{channel_id}/{msg.id}"
        if msg.chat.is_forum and msg.message_thread_id:
            url = f"t.me/c/{channel_id}/{msg.message_thread_id}/{msg.id}"

        btn = [[InlineKeyboardButton(text=text, url=url)]]
        if msg.from_user and text.lower() == "deleted":
            btn = [
                [InlineKeyboardButton(text="None", url=url)],
                [
                    InlineKeyboardButton(
                        text="From", url=f"tg://user?id={msg.from_user.id}"
                    )
                ],
            ]

        return InlineKeyboardMarkup(btn)


async def send_log(msg: Message, btn: InlineKeyboardMarkup) -> None:
    client = msg._client
    rep_to = None

    k, cache = (client.me.id, msg.chat.id, msg.id), bot.message_cache.store
    if k in cache.keys():
        rep_to, btn = ReplyParameters(message_id=cache.get(k)), None

    async def process(msg) -> None:
        async def download(file_id: str) -> io.BytesIO:
            client.max_concurrent_transmissions += 1
            try:
                return await client.download_media(message=file_id, in_memory=True)
            finally:
                client.max_concurrent_transmissions -= 1

        sent = None
        if msg.text:
            sent = await bot.send_message(
                chat_id=client.me.id,
                text=msg.text.html,
                reply_parameters=rep_to,
                reply_markup=btn,
            )
        else:
            obj = getattr(msg, msg.media.value)

            if msg.sticker:
                await bot.send_sticker(
                    chat_id=client.me.id,
                    sticker=obj.file_id,
                    reply_parameters=rep_to,
                    reply_markup=btn,
                )
            else:
                if isinstance(obj, Story):
                    msg = await client.get_stories(
                        story_sender_chat_id=obj.chat.id, story_ids=obj.id
                    )
                    obj = getattr(msg, msg.media.value)

                send_media = getattr(bot, f"send_{msg.media.value}")
                parameters = {
                    "chat_id": client.me.id,
                    "reply_parameters": rep_to,
                    "reply_markup": btn,
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
                    parameters.update(
                        {"thumb": await download(file_id=obj.thumbs[0].file_id)}
                    )

                for attr in ["view_once", "ttl_seconds"]:
                    parameters.pop(attr, None)

                sent = await send_media(**parameters)

        if sent:
            cache.update({k: sent.id})

    task = asyncio.create_task(coro=process(msg))

    try:
        await asyncio.wait_for(fut=task, timeout=15)
    except asyncio.TimeoutError:
        task.cancel()
    finally:
        del task


def fmt_secs(secs: int | float) -> str:
    if secs == 0:
        return "None"
    elif secs < 1e-3:
        return f"{secs * 1e6:.3f}".rstrip("0").rstrip(".") + " µs"
    elif secs < 1:
        return f"{secs * 1e3:.3f}".rstrip("0").rstrip(".") + " ms"
    return f"{secs:.3f}".rstrip("0").rstrip(".") + " s"


async def del_after(msg: Message, delay: int | float = 0.5) -> None:
    await asyncio.sleep(delay=delay)
    await msg.delete(revoke=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s - %(name)s: %(message)s"
    )
    logging.getLogger("pyrogram").setLevel(logging.ERROR)

    apps = []
    for k, v in enumerate(os.getenv("SESSION_STRING").split(), start=1):
        app = Client(
            name=str(k),
            session_string=v,
            parse_mode=bot.parse_mode,
            sleep_threshold=bot.sleep_threshold,
            max_message_cache_size=bot.max_message_cache_size,
            link_preview_options=bot.link_preview_options,
        )
        apps.append(app)

    async def main() -> None:
        async def start(client: Client) -> None:
            await client.start()

            logger = logging.getLogger(str(client.me.id))
            for group, (name, handler) in enumerate(cmds.items(), start=1):
                client.add_handler(handler=handler, group=group)
                logger.info(f"{name} Added")

            limit = await client.get_dialogs_count()
            async for _ in client.get_dialogs(limit=limit):
                pass

            await client.storage.save()

        await bot.start()
        logging.info("Client Helper Started")

        logging.info(f"Starting {len(apps)} App Clients")
        await asyncio.gather(*[asyncio.create_task(coro=start(client=i)) for i in apps])

    aiorun.logger.disabled = True
    aiorun.run(main(), loop=bot.loop)
