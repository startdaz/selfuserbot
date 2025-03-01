import asyncio

from pyrogram import Client
from pyrogram.enums import ParseMode

api_id = input("api_id: ")
api_hash = input("api_hash: ")
phone_number = input("phone_number: ")


async def generate() -> None:
    async with Client(
        name="app",
        api_id=int(api_id),
        api_hash=api_hash,
        phone_number=phone_number,
        parse_mode=ParseMode.DISABLED,
        in_memory=True,
        no_updates=True,
    ) as app:
        await app.storage.save()

        session_string = await app.export_session_string()
        await app.send_message(chat_id="me", text=session_string)


asyncio.run(generate())
