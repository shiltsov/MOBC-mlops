import fire


def test(string: str):
    print(string)


if __name__ == "__main__":
    fire.Fire({"test": test})
