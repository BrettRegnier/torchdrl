import torch
import torch.utils.data as data_utils
import gym
import time

from torchdrl.agents.Agent import Agent


# make a custom dataset that uses the env from my dragonboat.

class Supervised(Agent):
    def __init__(self,
        model, 
        optimizer, 
        criterion,
        scheduler=None,
        scheduler_step_type="epoch",
        trainset_loader=None,
        testset_loader=None,
        print_mode="simple",
        device="cpu"):

        assert isinstance(model, (torch.nn.Module,)), "A pytorch nn.Module is required"
        assert isinstance(optimizer, (torch.optim.Optimizer,)), "A pytorch optim.Optimizer is required, ex. Adam"
        # assert isinstance(criterion, (torch.nn.modules.loss._Loss)), "The criterion must be a pytorch Loss function" # Maybe remove? As people can have custom criterion
        self.SetTrainsetLoader(trainset_loader)
        self.SetTrainsetLoader(testset_loader)

        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._scheduler = scheduler
        self._scheduler_step_type = scheduler_step_type if scheduler_step_type == "epoch" else "step"
        self._trainset_loader = testset_loader
        self._print_mode = print_mode if print_mode == "simple" else "verbose"
        self._save_info = {}
        self._device = device 

        self._epochs = 0

    def Train(self, epochs=5, step_window=500):
        assert self._trainset_loader is not None, "No trainset loader provided, either during construction or runtime"
        self._model.train()

        self._epochs = epochs
        self._step_window = step_window

        self._total_train_size = len(self._trainset_loader) * self._epochs

        self._loss_list = []
        self._acc_list = []
        total_steps = 0
        total_loss = 0

        total_step = 1
        for epoch in range(self._epochs):
            steps = 1
            for samples, targets in self._trainset_loader:
                info = {}
                samples, targets = samples.to(self._device), targets.to(self._device)

                self._optimizer.zero_grad()

                predictions = self._model(samples)
                loss = self._criterion(predictions, targets)
                loss.backward()
                self._optimizer.step()

                total_loss += loss.item()

                if total_steps % self._step_window == 0:
                    pred = torch.argmax(predictions, dim=1)
                    correct_pred = pred.eq(targets)
                    accuracy = torch.mean(correct_pred.float())
                    avg_loss = total_loss / self._step_window

                    self._loss_list.append(avg_loss)
                    self._acc_list.append(accuracy)

                    info = {"epoch": epoch+1, "epochs": self._epochs, "total_steps": total_steps, "steps": steps, "avg_loss": avg_loss, "acc": accuracy}
                    yield info
                    
                    total_loss = 0

                if self._scheduler_step_type == "step":
                    self._scheduler.step()

                steps += 1
                total_steps += 1
            
            if self._scheduler_step_type == "epoch":
                self._scheduler.step()


    def TrainNoYield(self, trainset_loader, epochs=5, step_window=500):
        for step_window_info in self.Train(trainset_loader, epochs, step_window):
            epoch = step_window_info['epoch']
            epochs = step_window_info['epochs']
            total_steps = step_window_info['total_steps']
            steps = step_window_info['steps']
            avg_loss = step_window_info['avg_loss']
            accuracy = step_window_info['acc']

            self.PrintTrain(epoch, epochs, total_steps, step, avg_loss, accuracy)

    def PrintTrain(self, epoch, epochs, total_steps, step, avg_loss, accuracy):
        if self._print_mode == "simple":
            print('[Epoch {}/{}] Total Steps {} Steps {:<4} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, total_steps, steps, avg_loss, accuracy))
        else:
            total_space = 50
            progress = int(((total_steps-1) / self._total_train_size) * total_space)
            white_space = (total_space - progress) - 1

            print(f"\033[F [Epoch {epoch+1:>3}/{epochs:<3}] Total Steps: {total_steps:>7}, Steps: {steps:>6} [" + "=" * progress + ">" + " " * (white_space) + f"] Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.3f}")

    def Evaluate(self, epochs=1, max_steps=-1):
        """
        Evaluate the current model, given either a dataset loader or a gym environment.
        epochs: The number of tests to run
        max_steps: Exclusive to gym environments, gives the max allowed actions for that environment.
        """
        assert self._testset_loader is not None, "No testset loader provided or gym environment, either during construction or runtime"
        self._model.eval()

        test_info = {}
        if isinstance(self._testset_loader, (data_utils.DataLoader,)):
            # data loader TODO
            pass
        else:
            # env
            wins = 0
            losses = 0
            test_accuracy = 0
            total_steps = 0
            total_reward = 0

            start = time.time()
            for i in range(epochs):
                done = False
                steps = 0

                state = env.reset()
                last_info = None
                while steps != max_steps and not done:
                    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self._device)

                    pred = self._model(state_t)
                    action = torch.argmax(pred, dim=1)

                    next_state, reward, done, info = self._testset_loader.step(action)
                    state = next_state

                    total_reward += reward
                    last_info = info
                    steps += 1
                
                if 'win' in info:
                    win = info['win']
                    wins += 1 if win == 1 else 0
                    losses += 1 if win == 0 else 0

                total_steps += steps
            end = time.time()

            if 'win' in info:
                test_info['wins'] = wins
                test_info['losses'] = losses
                test_info['accuracy'] = round((float(wins) / float(wins+losses)) * 100, 2)

            test_info['total_steps'] = total_steps
            test_info['avg_reward'] = round(total_reward / epochs, 3)

        msg = ""
        for k, v in test_info.items():
            msg 

        return test_info

    def Act(self, state):
        return self._model(state)

    def GetAction(self, state):
        return self.Act(state)

    def Load(self, filepath):
        checkpoint = torch.load(filepath)

        self._model.load_state_dict(checkpoint['model'])
        self._optimizer.load_state_dict(checkpoint['optimizer'])
        
        if self._scheduler and 'scheduler' in checkpoint:
            self._scheduler.load_state_dict(checkpoint['scheduler'])

    def Save(self, folderpath, filename):
        folderpath = folderpath if folderpath[len(folderpath) - 1] == "/" else folderpath + "/"
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)

        self._save_info['model'] = self._model.state_dict()
        self._save_info['architecture'] = self._model.__str__()
        self._save_info['optimizer'] = self._optimizer.state_dict()
        self._save_info['learning_rate'] = self._optimizer.state_dict()['param_groups'][0]['initial_lr']
        self._save_info['epochs'] = self._epochs

        if self._scheduler:
            self._save_info['sched_step_size'] = self._scheduler.state_dict()['step_size']
            self._save_info['sched_gamma'] = self._scheduler.state_dict()['gamma']
            self._save_info['scheduler'] = self._scheduler.state_dict()

        torch.save(self._save_info, folderpath + filename)

    def SetTrainsetLoader(self, trainset_loader):
        assert trainset_loader is None or isinstance(trainset_loader, (data_utils.DataLoader,)), "The trainset_loader must be a pytorch DataLoader"
        self._trainset_loader = trainset_loader

    def SetTestsetLoader(self, testset_loader):        
        assert testset_loader is None isinstance(testset_loader, (data_utils.DataLoader, gym.Env)), "The testset_loader must be either a pytorch DataLoader or a gym Env"
        self._testset_loader = testset_loader


