import torch
from dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from critic import Critic
from generator import Generator
from utils import gradient_penalty
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import metrics
from scipy import signal

torch.backends.cudnn.benchmark = True
writer = SummaryWriter()
torch.cuda.empty_cache()

# form the train dataset
dataset_train = Dataset(
        root_a1=config.TRAIN_DIR+"/a1", root_a0=config.TRAIN_DIR+"/a0", #(a0 undamaged) (a1 damaged)
    )
# load the train dataset
loader_train = DataLoader(
        dataset_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
# if test dataset is used...
dataset_test = Dataset(
                root_a1=config.TEST_DIR+"/a1", root_a0=config.TEST_DIR+"/a0",
            )
loader_test = DataLoader(
                dataset_test,
                batch_size=4,
                shuffle=True,
                num_workers=config.NUM_WORKERS,
                pin_memory=True,
            )
# define the critics and generators
crit_a1 = Critic(channels=1).to(config.DEVICE)
crit_a0 = Critic(channels=1).to(config.DEVICE)
gen_a0 = Generator(channels=1).to(config.DEVICE)
gen_a1 = Generator(channels=1).to(config.DEVICE)
opt_crit = optim.AdamW(
        list(crit_a1.parameters()) + list(crit_a0.parameters()),
        lr=config.LEARNING_RATE_DISC,
        betas=(0.5, 0.999),
        amsgrad=True
    )
# set the optimizers
opt_gen = optim.AdamW(
        list(gen_a0.parameters()) + list(gen_a1.parameters()),
        lr=config.LEARNING_RATE_GEN,
        betas=(0.5, 0.999),
        amsgrad=True
    )
L1 = nn.L1Loss() #cycle & identity consistency loss
g_scaler = torch.cuda.amp.GradScaler()
c_scaler = torch.cuda.amp.GradScaler()

step_train=0
step_test=0
# Start training the model
for epoch in range(config.NUM_EPOCHS):
    loop_train = tqdm(loader_train, leave=True)
    for idx, (a0, a1) in enumerate(loop_train): 
        gen_a1.train()
        gen_a0.train()
        crit_a1.train()
        crit_a0.train()
        
        # fixing the dimensions of the tensors
        a0 = (a0.swapaxes(2, 1)).to(config.DEVICE, dtype=torch.float) 
        a1 = (a1.swapaxes(2, 1)).to(config.DEVICE, dtype=torch.float)

        # Train Critics
        #Critic
        with torch.cuda.amp.autocast():
            for _ in range(config.CRITIC_ITERATIONS):
                fake_a1 = gen_a1(a0)
                C_a1_real = crit_a1(a1).reshape(-1)
                C_a1_fake = crit_a1(fake_a1).reshape(-1)
                gp_C_a1 = gradient_penalty(crit_a1, a1, fake_a1, device=config.DEVICE)
                C_a1_loss = -(torch.mean(C_a1_real) - torch.mean(C_a1_fake)) + config.LAMBDA_GP * gp_C_a1

                fake_a0 = gen_a0(a1)             
                C_a0_real = crit_a0(a0).reshape(-1)
                C_a0_fake = crit_a0(fake_a0).reshape(-1)
                gp_C_a0 = gradient_penalty(crit_a0, a0, fake_a0, device=config.DEVICE)
                C_a0_loss = -(torch.mean(C_a0_real) - torch.mean(C_a0_fake)) + config.LAMBDA_GP * gp_C_a0

                # put it together
                C_loss = (C_a1_loss + C_a0_loss)/2
                
                opt_crit.zero_grad(set_to_none=True)
                c_scaler.scale(C_loss).backward(retain_graph=True)
                c_scaler.step(opt_crit)
                c_scaler.update()

        # Train Generators
            # adversarial loss for both generators
        with torch.cuda.amp.autocast():
            C_a1_fake = crit_a1(fake_a1).reshape(-1)
            C_a0_fake = crit_a0(fake_a0).reshape(-1)
            G_a1_loss = -torch.mean(C_a1_fake)
            G_a0_loss = -torch.mean(C_a0_fake)

            # cycle loss
            cycle_a0 = gen_a0(fake_a0) 
            cycle_a1 = gen_a1(fake_a0)
            cycle_a0_loss = L1(a0, cycle_a0)
            cycle_a1_loss = L1(a1, cycle_a1)
    
            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_a0 = gen_a0(a0)
            identity_a1 = gen_a1(a1)
            identity_a0_loss = L1(a0, identity_a0)
            identity_a1_loss = L1(a1, identity_a1)
                
            # frequency loss
            freqs_a1 = torch.fft.fft(a1)
            freqs_a0 = torch.fft.fft(a0)
    
            freqs_fake_a0 = torch.fft.fft(fake_a0.to(dtype=torch.float))
            freqs_a0_loss = (L1(freqs_a0.abs(), freqs_fake_a0.abs()) 
                                + L1(freqs_a0.angle(), freqs_fake_a0.angle()))
            freqs_fake_a1 = torch.fft.fft(fake_a1.to(dtype=torch.float))
            freqs_a1_loss = (L1(freqs_a1.abs(), freqs_fake_a1.abs()) 
                                + L1(freqs_a1.angle(), freqs_fake_a1.angle()))
                # add all together
            G_loss = (
                      G_a1_loss
                    + G_a0_loss
                    + cycle_a0_loss * config.LAMBDA_CYCLE
                    + cycle_a1_loss * config.LAMBDA_CYCLE
                    + identity_a1_loss * config.LAMBDA_IDENTITY
                    + identity_a0_loss * config.LAMBDA_IDENTITY
                    + freqs_a0_loss * config.LAMBDA_FREQS
                    + freqs_a1_loss * config.LAMBDA_FREQS
            )
            
        opt_gen.zero_grad(set_to_none=True)
        g_scaler.scale(G_loss).backward(retain_graph=True)
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        with torch.no_grad():
            # write it to Tensorboard (see how to setup and use the tensorboard with tpytorch)
            gen_a0.eval()
            gen_a1.eval()
            
            # Below, removing the gpu, converting to arrays etc. I'm sure there are better ways to these.
            a1_cpu = a1.cpu()
            a1_cpu_np = np.array(a1_cpu)
            a1_cpu_np0 = a1_cpu_np[0]
            a1_cpu_np0 = np.squeeze(a1_cpu_np0)
            fake_a1_cpu = fake_a1.cpu()
            fake_a1_cpu_np = np.array(fake_a1_cpu.detach())
            fake_a1_cpu_np0 = fake_a1_cpu_np[0]
            fake_a1_cpu_np0 = np.squeeze(fake_a1_cpu_np0)
           
            a0_cpu = a0.cpu()
            a0_cpu_np = np.array(a0_cpu)
            a0_cpu_np0 = a0_cpu_np[0]
            a0_cpu_np0 = np.squeeze(a0_cpu_np0)
            fake_a0_cpu = fake_a0.cpu()
            fake_a0_cpu_np = np.array(fake_a0_cpu.detach())
            fake_a0_cpu_np0 = fake_a0_cpu_np[0]
            fake_a0_cpu_np0 = np.squeeze(fake_a0_cpu_np0)
           
            fid_a1_fakea1 = metrics.calculate_fid(a1_cpu_np0,fake_a1_cpu_np0)
            fid_a0_fakea0 = metrics.calculate_fid(a0_cpu_np0,fake_a0_cpu_np0)
            mscohere_a1_fakea1 = np.mean((signal.coherence(a1_cpu_np0,fake_a1_cpu_np0
                                                                 , 256, nperseg = 256))[1])
            mscohere_a0_fakea0 = np.mean((signal.coherence(a0_cpu_np0,fake_a0_cpu_np0
                                                                 , 256, nperseg = 256))[1])
            
            writer.add_scalar("fid_a1_fakea1", fid_a1_fakea1, global_step=step_train)
            writer.add_scalar("fid_a0_fakea0", fid_a0_fakea0, global_step=step_train)
            writer.add_scalar("mscohere_a1_fakea1", mscohere_a1_fakea1, global_step=step_train)
            writer.add_scalar("mscohere_a0_fakea0", mscohere_a0_fakea0, global_step=step_train)
            writer.add_scalar("C_loss", C_loss, global_step=step_train)
            writer.add_scalar("G_loss", G_loss, global_step=step_train)

            step_train += 1
            loop_train.set_description(f"Epoch_train {epoch}")
             

# Save the model        
torch.save(gen_a1, 'gen_a1.pkl')
torch.save(gen_a0, 'gen_a0.pkl')

#count model parameters
def count_parameters(network1,network2,network3,network4):
    return(sum(p.numel() for p in network1.parameters() if p.requires_grad)+
             sum(p.numel() for p in network2.parameters() if p.requires_grad)+
             sum(p.numel() for p in network3.parameters() if p.requires_grad)+
             sum(p.numel() for p in network4.parameters() if p.requires_grad))

total_params = count_parameters(gen_a1, gen_a0, crit_a1, crit_a0)


    


