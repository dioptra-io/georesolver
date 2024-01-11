import time
import asyncio
import httpx
from loguru import logger


async def get_raw_vps(url: str = "https://atlas.ripe.net/api/v2/probes/"):
    """get request url atlas endpoint"""
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp = resp.json()

        while resp["next"]:
            for vp in resp["results"]:
                yield vp

            resp = await client.get(url)
            resp = resp.json()

            await asyncio.sleep(0.1)


async def main():
    async for vp in get_raw_vps():
        logger.info(vp)


if __name__ == "__main__":
    s = time.perf_counter()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")
