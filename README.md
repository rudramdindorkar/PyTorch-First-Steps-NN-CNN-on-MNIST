# PyTorch-First-Steps-NN-CNN-on-MNIST
My first hands-on PyTorch project, built while learning deep learning fundamentals. Implemented two models from scratch on the MNIST digit classification dataset.


What I Built
Model 1 — Plain Neural Network
A simple fully connected network.
Input (784) → Linear → ReLU → Linear → Output (10)

Model 2 — Convolutional Neural Network
A CNN that learns spatial features from raw images.
Input (1, 28, 28)
→ Conv2d(1, 32, 3) → ReLU → MaxPool2d(2)
→ Conv2d(32, 64, 3) → ReLU → MaxPool2d(2)
→ Flatten → Linear(1600, 128) → ReLU → Linear(128, 10)

Results
Accuracy: Neural Network~97%  CNN~99%


What I Learned

How PyTorch's nn.Module works and why forward() is needed

The training loop: zero_grad → forward → loss → backward → step

Why zero_grad() is critical (PyTorch accumulates gradients)

How convolutions shrink spatial dimensions: output = input - kernel + 1

What MaxPooling does and why it helps generalization

Difference between model.train() and model.eval()

Why torch.no_grad() is used during inference



