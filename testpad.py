from utils import load_array_multiprocess


def main():
    model = load_array_multiprocess()
    return model


model = main()


for k, v in model.items():
    print(k, v)
    break
