import torch
import torch.utils.data as data_utils
import os
import gym
import time

from torchdrl.agents.Agent import Agent



# make a custom dataset that uses the env from my dragonboat.

# TODO add testing frequency
class SupervisedAgent(Agent):
    def __init__(self,
        model, 
        optimizer, 
        criterion,
        trainset_loader,
        testset_loader,
        scheduler=None,
        scheduler_step_type="epoch",
        step_window=500,
        evaluate_per_print=False,
        evaluate_amt=1,
        record_chart_metrics=False,
        printing_type="step",
        print_mode="simple",
        device="cpu"
    ):

        assert isinstance(model, (torch.nn.Module,)), "A pytorch nn.Module is required"
        assert isinstance(optimizer, (torch.optim.Optimizer,)), "A pytorch optim.Optimizer is required, ex. Adam"
        # assert isinstance(criterion, (torch.nn.modules.loss._Loss)), "The criterion must be a pytorch Loss function" # Maybe remove? As people can have custom criterion
        self.SetTrainsetLoader(trainset_loader)
        self.SetTestsetLoader(testset_loader)

        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._scheduler = scheduler
        self._scheduler_step_type = scheduler_step_type if scheduler_step_type == "epoch" else "step"
        self._step_window = step_window
        self._evaluate_per_print = evaluate_per_print if evaluate_per_print == True else False
        self._evaluate_amt = evaluate_amt
        self._record_chart_metrics = record_chart_metrics if record_chart_metrics == True else False
        self._printing_type = printing_type if printing_type == "epoch" else "step"
        self._print_mode = print_mode if print_mode == "simple" else "verbose"
        self._save_info = {}
        self._device = device 

        self._epochs = 0
        self._epoch = 0

        # TODO make generic some way maybe?
        self._chart_metrics = {
            "train": 
            {
                "epoch": [],
                "loss": [],
                "accuracy": [],
            },
            "test":
            {
                "epoch": [],
                "loss": [],
                "accuracy": [],
            }
        }

    def Train(self, epochs=5):
        assert self._trainset_loader is not None, "No trainset loader provided, either during construction or runtime"
        self._model.train()

        self._epochs = epochs
        self._epoch = 0

        self._total_train_size = len(self._trainset_loader) * self._epochs

        self._loss_list = []
        self._acc_list = []
        total_steps = 0
        total_loss = 0

        predictions = None
        samples = None
        targets = None
        while self._epoch < self._epochs:
            self._epoch += 1
            steps = 0
            for samples, targets in self._trainset_loader:
                steps += 1
                total_steps += 1

                train_info = {}
                samples = samples.to(self._device)
                targets = targets.to(self._device)

                self._optimizer.zero_grad()

                predictions = self._model(samples)
                loss = self._criterion(predictions, targets)
                loss.backward()
                self._optimizer.step()

                total_loss += loss.item()

                if total_steps % self._step_window == 0 and self._printing_type == "step":
                    avg_loss, accuracy = self.TrainPredictionInfo(predictions, targets, total_loss)
                    total_loss = 0
                    train_info = {"epoch": self._epoch, "epochs": self._epochs, "total_steps": total_steps, "steps": steps, "avg_loss": avg_loss, "acc": accuracy}

                    test_info = {}
                    test_msg = ""
                    if self._evaluate_per_print:
                        test_info, test_msg = self.Evaluate(self._evaluate_amt, 20)
                    train_msg = self.TrainMessage(self._epoch, epochs, total_steps, steps, avg_loss, accuracy)
                    
                    yield train_info, train_msg, test_info, test_msg
                    
                if self._scheduler_step_type == "step" and self._scheduler:
                    self._scheduler.step()

            if self._printing_type == "epoch":
                avg_loss, accuracy = self.TrainPredictionInfo(predictions, targets, total_loss)
                total_loss = 0
                train_info = {"epoch": self._epoch, "epochs": self._epochs, "total_steps": total_steps, "steps": steps, "avg_loss": avg_loss, "acc": accuracy} 

                test_info = {}
                test_msg = ""
                if self._evaluate_per_print:
                    test_info, test_msg = self.Evaluate(self._evaluate_amt, 20)
                train_msg = self.TrainMessage(self._epoch, epochs, total_steps, steps, avg_loss, accuracy)

                yield train_info, train_msg, test_info, test_msg

            if self._scheduler_step_type == "epoch" and self._scheduler:
                self._scheduler.step()

    def TrainPredictionInfo(self, predictions, targets, total_loss):                    
        pred = torch.argmax(predictions, dim=1)
        correct_pred = pred.eq(targets)
        accuracy = torch.mean(correct_pred.float())
        avg_loss = total_loss / self._step_window

        if self._record_chart_metrics:
            self._chart_metrics['epoch'] = self._epoch
            self._chart_metrics['loss'] = avg_loss
            self._chart_metrics['accuracy'] = accuracy

        return (avg_loss, accuracy)

    def TrainNoYield(self, epochs=5):
        for _, train_msg, _, test_msg in self.Train(epochs):
            print(train_msg, test_msg)

    def TrainMessage(self, epoch, epochs, total_steps, steps, avg_loss, accuracy):
        msg = ""
        if self._print_mode == "simple":
            out_of = str(epoch) + "/" + str(epochs)
            msg = f"[Epoch {out_of:>6}] Total Steps: {total_steps:>7}, Steps: {steps:>6} -> Train Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%"
        else:
            total_space = 50
            progress = int(((total_steps-1) / self._total_train_size) * total_space)
            white_space = (total_space - progress) - 1

            out_of = str(epoch) + "/" + str(epochs)
            msg = f"[Epoch {out_of:>6}] Total Steps: {total_steps:>7}, Steps: {steps:>6} [" + "=" * progress + ">" + " " * (white_space) + f"] Train Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%"

        return msg

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
            total_accuracy = 0
            total_loss = 0
            num_items = 0
            for samples, targets in self._testset_loader:
                samples = samples.to(self._device)
                targets = targets.to(self._device)

                predictions = self._model(samples)
                total_loss += self._criterion(predictions, targets)

                predictions = torch.argmax(predictions, dim=1)
                corrects = predictions.eq(targets)
                total_accuracy += torch.mean(corrects.float())
                num_items += 1

            accuracy = total_accuracy / num_items
            loss = total_loss / num_items
            test_info['accuracy'] = accuracy

            if self._record_chart_metrics:
                self._chart_metrics['epoch'] = self._epoch
                self._chart_metrics['loss'] = loss
                self._chart_metrics['accuracy'] = accuracy
        else:
            # env
            wins = 0
            loses = 0
            total_steps = 0
            total_reward = 0
            accuracy = None

            start = time.time()
            for i in range(epochs):
                done = False
                steps = 0

                state = self._testset_loader.reset()
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
                    loses += 1 if win == 0 else 0

                total_steps += steps
            end = time.time()

            test_info['total_steps'] = total_steps
            test_info['avg_steps'] = round(total_steps / epochs, 1)
            test_info['avg_reward'] = round(total_reward / epochs, 3)

            if 'win' in info:
                test_info['wins'] = wins
                test_info['loses'] = loses
                accuracy = float(wins) / float(wins+loses)
                test_info['accuracy'] = accuracy
            


        test_info['epochs'] = epochs

        self._save_info['test_info'] = test_info
        test_msg = self.TestMessage(test_info)

        return test_info, test_msg

    def TestMessage(self, test_info):

        msg = "Test: "
        if 'accuracy' in test_info:
            msg += f"Accuracy: {test_info['accuracy']*100:.2f}% "
        if self._print_mode != "simple":
            if 'avg_steps' in test_info:
                msg += f"Average Steps: {test_info['avg_steps']}"
            if 'wins' in test_info:
                msg += f"W: {test_info['wins']} "
            if 'loses' in test_info:
                msg += f"L: {test_info['loses']} "
        return msg

    def GetAction(self, state):
        return self.Act(state)

    def Act(self, state):
        return self._model(state)

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

        self._save_info['epochs'] = self._epochs
        
        self._save_info['charts'] = self._chart_metrics

        if self._scheduler:
            self._save_info['sched_step_size'] = self._scheduler.state_dict()['step_size']
            self._save_info['sched_gamma'] = self._scheduler.state_dict()['gamma']
            self._save_info['scheduler'] = self._scheduler.state_dict()
            self._save_info['learning_rate'] = self._optimizer.state_dict()['param_groups'][0]['initial_lr']
        else:
            self._save_info['learning_rate'] = self._optimizer.state_dict()['param_groups'][0]['lr']

        torch.save(self._save_info, folderpath + filename)

    def SetTrainsetLoader(self, trainset_loader):
        assert isinstance(trainset_loader, (data_utils.DataLoader,)), "The trainset_loader must be a pytorch DataLoader"
        self._trainset_loader = trainset_loader

    def SetTestsetLoader(self, testset_loader):        
        assert isinstance(testset_loader, (data_utils.DataLoader, gym.Env)), "The testset_loader must be either a pytorch DataLoader or a gym Env"
        self._testset_loader = testset_loader


