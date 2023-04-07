from torch.optim.lr_scheduler import MultiStepLR, LinearLR
from numpy import arange


class SchedulerFactory:
    @staticmethod
    def create_scheduler(optimizer, scheduler_name, base_scheduler_name=None,
                         epoch_start_cluster=0):
        if scheduler_name == "default":
            return DefaultScheduler(optimizer)
        elif scheduler_name == "defaultwithlongerdecay":
            return DefaultWithLongerDecayScheduler(optimizer)
        elif scheduler_name == "defaultwithlongerdecay2":
            return DefaultWithLongerDecay2Scheduler(optimizer)
        elif scheduler_name == "defaultwithlongerwarmup":
            return DefaultWithLongerWarmUpScheduler(optimizer)
        elif scheduler_name == "myCustomScheduler":
            base_scheduler_class = SchedulerFactory.create_scheduler(
                optimizer,
                base_scheduler_name).__class__
            return CustomScheduler(optimizer=optimizer,
                                   base_scheduler_class=base_scheduler_class)
        else:
            raise Exception("Scheduler not implemented.")


class CustomLR(LinearLR):
    def __init__(self, optimizer, total_epochs, drop_factor,
                 drop_interval, drop_duration, last_epoch=-1,
                 base_conds=None,
                 base_scheduler=None):

        self.total_epochs = total_epochs
        self.drop_factor = drop_factor
        self.drop_interval = drop_interval
        self.drop_duration = drop_duration
        self.optimizer = optimizer
        super(CustomLR, self).__init__(optimizer=optimizer,
                                       start_factor=0.001,
                                       end_factor=1.0,
                                       total_iters=10,
                                       last_epoch=last_epoch
                                       )

    def get_lr(self):
        linear_progress = self.last_epoch / self.total_epochs
        linear_lr = [base_lr * (1 - linear_progress) for base_lr in self.base_lrs]

        if self.last_epoch % self.drop_interval < self.drop_duration:
            self.step()
            return [lr / self.drop_factor for lr in linear_lr]
        else:
            return linear_lr


class BaseScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.schedulers = []
        self.conds = []

    def step(self, epoch):
        for scheduler, cond in zip(self.schedulers, self.conds):
            if cond(epoch):
                scheduler.step()


class DefaultScheduler(BaseScheduler):
    def __init__(self, optimizer):
        super().__init__(optimizer)
        self.schedulers = [
            self._default_warm_up_scheduler(),
            self._default_scheduler()
        ]
        self.conds = [
            lambda epoch: epoch < 10,
            lambda epoch: True
        ]

    def _default_scheduler(self):
        return MultiStepLR(optimizer=self.optimizer,
                           milestones=arange(30, 100), gamma=0.1 ** (1 / 70))

    def _default_warm_up_scheduler(self):
        return LinearLR(optimizer=self.optimizer,
                        start_factor=0.001, end_factor=1.0,
                        total_iters=10, last_epoch=-1)


class DefaultWithLongerDecayScheduler(DefaultScheduler):
    def __init__(self, optimizer):
        super().__init__(optimizer)
        self.schedulers[1] = self._longer_decay_scheduler()
        self.conds[1] = lambda epoch: 30 <= epoch < 150

    def _longer_decay_scheduler(self):
        return LinearLR(optimizer=self.optimizer, start_factor=1.0,
                        end_factor=0.3, total_iters=120, last_epoch=-1)


class DefaultWithLongerDecay2Scheduler(DefaultScheduler):
    def __init__(self, optimizer):
        super().__init__(optimizer)
        self.schedulers[1] = self._default_scheduler_shifted2()

    def _default_scheduler_shifted2(self):
        return MultiStepLR(optimizer=self.optimizer,
                           milestones=arange(30, 150), gamma=0.3 ** (1 / 120))


class DefaultWithLongerWarmUpScheduler(DefaultScheduler):
    def __init__(self, optimizer):
        super().__init__(optimizer)
        self.schedulers[0] = self._longer_warm_up_scheduler()
        self.schedulers[1] = self._default_scheduler_shifted()
        self.conds[0] = lambda epoch: epoch < 50

    def _longer_warm_up_scheduler(self):
        return LinearLR(optimizer=self.optimizer, start_factor=0.001,
                        end_factor=1.0, total_iters=50, last_epoch=-1)

    def _default_scheduler_shifted(self):
        return MultiStepLR(optimizer=self.optimizer, milestones=arange(80, 170),
                           gamma=0.1 ** (1 / 70))


class CustomScheduler(BaseScheduler):
    def __init__(self, optimizer, base_scheduler_class, total_epochs=100,
                 drop_factor=100,
                 drop_interval=10,
                 drop_duration=5
                 ):
        super().__init__(optimizer)

        self.base_scheduler = base_scheduler_class(optimizer)

        custom_scheduler = CustomLR(optimizer, total_epochs, drop_factor,
                                    drop_interval, drop_duration,
                                    base_conds=self.base_scheduler.schedulers,
                                    base_scheduler=self.base_scheduler.conds)

        self.schedulers.append(custom_scheduler)
        self.conds.append(lambda epoch: True)
