import torch

# Sprawdzenie wersji
print(f"Wersja PyTorch: {torch.__version__}")

# Sprawdzenie, czy widzi kartę graficzną
if torch.cuda.is_available():
    print(f"Wykryto GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU niewykryte - obliczenia na CPU.")

# Test tensorów i autogradu (niezbędne do równań różniczkowych w sieciach)
x = torch.tensor([[2.0, 3.0]], requires_grad=True)
y = x ** 2
out = y.mean()
out.backward()

print(f"Gradient x (powinien wynosić [2.0, 3.0]): {x.grad}")