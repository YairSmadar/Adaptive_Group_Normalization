from torch.optim.lr_scheduler import MultiStepLR, LinearLR, CyclicLR
from abc import ABC, abstractmethod
from global_vars import args


class BaseScheduler(ABC):
    @abstractmethod
    def get_scheduler(self, optimizer):
        pass

    @abstractmethod
    def condition(self, epoch):
        pass


class DefaultWarmUpScheduler(BaseScheduler):
    NAME = "defaultwarmup"

    def get_scheduler(self, optimizer):
        return LinearLR(optimizer=optimizer, start_factor=0.001, end_factor=1.0,
                        total_iters=10, last_epoch=-1)

    def condition(self, epoch):
        return epoch < 10


class DefaultScheduler(BaseScheduler):
    NAME = "deafultmultistep"

    def get_scheduler(self, optimizer):
        return MultiStepLR(optimizer=optimizer, milestones=range(30, 100),
                           gamma=0.1 ** (1 / 70))

    def condition(self, epoch):
        return True


class MultiStepLRScheduler(BaseScheduler):
    NAME = "multisteplr"

    def get_scheduler(self, optimizer):
        return MultiStepLR(optimizer=optimizer, milestones=[30,70,90],
                           gamma=0.5)

    def condition(self, epoch):
        return True


class ExtraMultiStepLRScheduler(BaseScheduler):
    NAME = "extramultisteplr"

    def get_scheduler(self, optimizer):
        return MultiStepLR(optimizer=optimizer, milestones=range(10,90,5),
                           gamma=0.95)

    def condition(self, epoch):
        return True


class Triangular2Scheduler(BaseScheduler):
    NAME = "triangular2"

    def get_scheduler(self, optimizer):
        return CyclicLR(optimizer, base_lr=args.lr*0.1, max_lr=args.lr,
                        step_size_up=5, step_size_down=10, mode='triangular2')

    def condition(self, epoch):
        return True  # Always apply this scheduler.


class SchedulerFactory:
    def __init__(self):
        self.scheduler_classes = {
            DefaultWarmUpScheduler.NAME: DefaultWarmUpScheduler,
            DefaultScheduler.NAME: DefaultScheduler,
            MultiStepLRScheduler.NAME: MultiStepLRScheduler,
            ExtraMultiStepLRScheduler.NAME: ExtraMultiStepLRScheduler,
            Triangular2Scheduler.NAME: Triangular2Scheduler
        }

    def get_scheduler(self, scheduler_name):
        scheduler_class = self.scheduler_classes.get(scheduler_name)
        if scheduler_class is None:
            raise KeyError(f'There is no scheduler name {scheduler_name}')
        return scheduler_class()


class SchedulerManager:
    def __init__(self, scheduler_name):
        self.schedulers = []
        self.factory = SchedulerFactory()
        self.scheduler_name = scheduler_name

    def add_scheduler(self, scheduler_name, optimizer):
        scheduler = self.factory.get_scheduler(scheduler_name)
        scheduler_instance = scheduler.get_scheduler(optimizer)
        self.schedulers.append((scheduler_instance, scheduler.condition))

    def set_schedulers(self, optimizer):
        if self.scheduler_name == 'default':
            self.add_scheduler(DefaultWarmUpScheduler.NAME, optimizer)
            self.add_scheduler(DefaultScheduler.NAME, optimizer)
        elif self.scheduler_name == 'multisteplr':
            self.add_scheduler(MultiStepLRScheduler.NAME, optimizer)
        elif self.scheduler_name == 'triangular2':
            self.add_scheduler(Triangular2Scheduler.NAME, optimizer)
        elif self.scheduler_name == 'extramultisteplr':
            self.add_scheduler(ExtraMultiStepLRScheduler.NAME, optimizer)
        else:
            raise KeyError(f'There is no scheduler name {self.scheduler_name}')

    def get_scheduler_name(self):
        return self.scheduler_name

    def schedulers_step(self, epoch):
        for scheduler, cond in self.schedulers:
            if cond(epoch):
                scheduler.step()
