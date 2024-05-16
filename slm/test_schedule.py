
#%% plot warmup_cosine_lambda

import math   
def warmup_cosine_lambda(current_step, epoch_steps= 10, warmup_epochs=1, annealing_epochs=10, min_lr_ratio = 0.1):
            # Assumes 1 epoch = len(train_dataloader) steps
            num_warmup_steps = epoch_steps * warmup_epochs
            num_annealing_steps = epoch_steps * annealing_epochs
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            elif current_step > num_warmup_steps:
                progress = (current_step - num_warmup_steps) / num_annealing_steps
                return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
import matplotlib.pyplot as plt

epoch_steps = 10
warmup_epochs = 1
annealing_epochs = 10
min_lr_ratio = 0.1
steps = 100
lrs = [warmup_cosine_lambda(i, epoch_steps, warmup_epochs, annealing_epochs, min_lr_ratio) for i in range(steps)]
plt.plot(lrs)
plt.show()
# %%
