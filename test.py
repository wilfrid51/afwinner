import asyncio
from sat import SATTask
from abd import ABDTask
from ded import DEDTask

async def main():
    while True:
        sat = SATTask()
        sat_challenge = await sat.generate()
        # print(sat_challenge)
        # print(f"[SAT]\n{sat_challenge.prompt}")
        response = "x1=True, x2=False, x3=True, x4=False, x5=True"
        sat_score = await sat.evaluate(response, sat_challenge)
        print(f"[SAT] Score: {sat_score}")

        abd = ABDTask()
        abd_challenge = await abd.generate()
        # print(abd_challenge)
        # print(f"[ABD]\n{abd_challenge.prompt}")
        response = "<INPUT>1 2 3</INPUT>"
        abd_score = await abd.evaluate(response, abd_challenge)
        print(f"[ABD] Score: {abd_score}")

        ded = DEDTask()
        ded_challenge = await ded.generate()
        # print(ded_challenge)
        # print(f"[DED]\n{ded_challenge.prompt}")
        response = """
        def add(a, b):
            return a + b
        if __name__ == '__main__':
            result = add(1, 2)
            print(result)
        """
        ded_score = await ded.evaluate(response, ded_challenge)
        print(f"[DED] Score: {ded_score}")
        await asyncio.sleep(1)
        print("--------------------------------")

if __name__ == "__main__":
    asyncio.run(main())