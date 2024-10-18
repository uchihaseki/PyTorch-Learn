import torch

outputs = torch.tensor([[0.1,0.2,0.5],
                        [0.1,0.6,0.3],
                        [0.2,0.2,0.2]]
                       )

print(outputs.argmax(0))
