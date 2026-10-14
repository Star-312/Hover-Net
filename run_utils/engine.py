import tqdm
from enum import Enum


####
class Events(Enum):
    EPOCH_STARTED = "epoch_started"
    EPOCH_COMPLETED = "epoch_completed"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STARTED = "started"
    COMPLETED = "completed"
    EXCEPTION_RAISED = "exception_raised"


####
class State(object):
    """An object that is used to pass internal and user-defined state between event handlers."""
    """用于在事件处理器之间传递内部和用户定义状态的对象。"""

    def __init__(self):
        # settings propagated from config
        # 从配置传递的设置
        self.logging = None
        self.log_dir = None
        self.log_info = None

        # internal variable
        # 内部变量
        self.curr_epoch_step = 0  # current step in epoch
        # 当前 epoch 中的步数
        self.curr_global_step = 0  # current global step
        # 当前全局步数
        self.curr_epoch = 0  # current global epoch
        # 当前全局 epoch

        # TODO: [LOW] better document this
        # TODO: [低优先级] 更好地记录这个
        # for outputing value that will be tracked per step
        # 输出将按步跟踪的值
        # "scalar" will always be printed out and added to the tensorboard
        # "scalar" 将始终打印并添加到 tensorboard
        # "images" will need dedicated function to process and added to the tensorboard
        # "images" 需要专门的函数处理并添加到 tensorboard

        # ! naming should match with types supported for serialize
        # ! 命名应与 serialize 支持的类型匹配
        # TODO: Need way to dynamically adding new types
        # TODO: 需要动态添加新类型的方式
        self.tracked_step_output = {
            "scalar": {},  # type : {variable_name : variablee_value}
            # 类型 : {变量名 : 变量值}
            "image": {},
        }
        # TODO: find way to known which method bind/interact with which value
        # TODO: 找到知道哪个方法绑定/交互哪个值的方式

        self.epoch_accumulated_output = {}  # all output of the current epoch
        # 当前 epoch 的所有输出

        # TODO: soft reset for pertain variable for N epochs
        # TODO: 对 N 个 epoch 的持久变量进行软重置
        self.run_accumulated_output = []  # of run until reseted
        # 运行直到重置

        # holder for output returned after current runstep
        # 当前 runstep 后返回的输出的持有者
        # * depend on the type of training i.e GAN, the updated accumulated may be different
        # * 取决于训练类型，例如 GAN，更新的累积值可能不同
        self.step_output = None

        self.global_state = None
        return

    def reset_variable(self):
        # type : {variable_name : variable_value}
        # 类型 : {变量名 : 变量值}
        self.tracked_step_output = {k: {} for k in self.tracked_step_output.keys()}

        # TODO: [CRITICAL] refactor this
        # TODO: [关键] 重构这个
        if self.curr_epoch % self.pertain_n_epoch_output == 0:
            self.run_accumulated_output = []

        self.epoch_accumulated_output = {}

        # * depend on the type of training i.e GAN, the updated accumulated may be different
        # * 取决于训练类型，例如 GAN，更新的累积值可能不同
        self.step_output = None  # holder for output returned after current runstep
        # 当前 runstep 后返回的输出的持有者
        return


####
class RunEngine(object):
    """
    TODO: Include docstring
    """
    """
    TODO: 添加文档字符串
    """

    def __init__(
        self,
        engine_name=None,
        dataloader=None,
        run_step=None,
        run_info=None,
        log_info=None,  # TODO: refactor this with trainer.py
        # TODO: 用 trainer.py 重构这个
    ):

        # * auto set all input as object variables
        # * 自动将所有输入设置为对象变量
        self.engine_name = engine_name
        self.run_step = run_step
        self.dataloader = dataloader

        # * global variable/object holder shared between all event handler
        # * 所有事件处理器之间共享的全局变量/对象持有者
        self.state = State()
        # * check if correctly referenced, not new copies
        # * 检查是否正确引用，而不是新副本
        self.state.attached_engine_name = engine_name  # TODO: redundant?
        # TODO: 多余吗？
        self.state.run_info = run_info
        self.state.log_info = log_info
        self.state.batch_size = dataloader.batch_size

        # TODO: [CRITICAL] match all the mechanism outline with opt
        # TODO: [关键] 将所有机制与 opt 匹配
        self.state.pertain_n_epoch_output = 1 if engine_name == "valid" else 1

        self.event_handler_dict = {event: [] for event in Events}

        # TODO: think about this more
        # TODO: 再想想这个
        # to share global state across a chain of RunEngine such as
        # 在 RunEngine 链之间共享全局状态，例如
        # from the engine for training to engine for validation
        # 从训练引擎到验证引擎

        #
        self.terminate = False
        return

    def __reset_state(self):
        # TODO: think about this more, looks too redundant
        # TODO: 再想想这个，看起来太冗余了
        new_state = State()
        new_state.attached_engine_name = self.state.attached_engine_name
        new_state.run_info = self.state.run_info
        new_state.log_info = self.state.log_info
        self.state = new_state
        return

    def __trigger_events(self, event):
        for callback in self.event_handler_dict[event]:
            callback.run(self.state, event)
            # TODO: exception and throwing error with name or sthg to allow trace back
            # TODO: 异常并抛出带有名称的错误，以便追溯
        return

    # TODO: variable to indicate output dependency between handler !
    # TODO: 指示处理器之间输出依赖关系的变量！
    def add_event_handler(self, event_name, handler):
        self.event_handler_dict[event_name].append(handler)

    # ! Put into trainer.py ?
    # ! 放入 trainer.py？
    def run(self, nr_epoch=1, shared_state=None, chained=False):

        # TODO: refactor this
        # TODO: 重构这个
        if chained:
            self.state.curr_epoch = 0
        self.state.global_state = shared_state

        while self.state.curr_epoch < nr_epoch:
            self.state.reset_variable()  # * reset all EMA holder per epoch
            # * 每个 epoch 重置所有 EMA 持有者

            if not chained:
                print("----------------EPOCH %d" % (self.state.curr_epoch + 1))

            self.__trigger_events(Events.EPOCH_STARTED)

            pbar_format = (
                "Processing: |{bar}| "
                "{n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            )
            if self.engine_name == "train":
                pbar_format += (
                    "Batch = {postfix[1][Batch]:0.5f}|" "EMA = {postfix[1][EMA]:0.5f}"
                )
                # * changing print char may break the bar so avoid it
                # * 更改打印字符可能会破坏进度条，所以避免这样做
                pbar = tqdm.tqdm(
                    total=len(self.dataloader),
                    leave=True,
                    initial=0,
                    bar_format=pbar_format,
                    ascii=True,
                    postfix=["", dict(Batch=float("NaN"), EMA=float("NaN"))],
                )
            else:
                pbar = tqdm.tqdm(
                    total=len(self.dataloader),
                    leave=True,
                    bar_format=pbar_format,
                    ascii=True,
                )

            for data_batch in self.dataloader:
                self.__trigger_events(Events.STEP_STARTED)

                step_run_info = [
                    self.state.run_info,
                    {
                        "epoch": self.state.curr_epoch,
                        "step": self.state.curr_global_step,
                    },
                ]
                step_output = self.run_step(data_batch, step_run_info)
                self.state.step_output = step_output

                self.__trigger_events(Events.STEP_COMPLETED)
                self.state.curr_global_step += 1
                self.state.curr_epoch_step += 1

                if self.engine_name == "train":
                    pbar.postfix[1]["Batch"] = step_output["EMA"]["overall_loss"]
                    pbar.postfix[1]["EMA"] = self.state.tracked_step_output["scalar"][
                        "overall_loss"
                    ]
                pbar.update()
            pbar.close()  # to flush out the bar before doing end of epoch reporting
            # 在进行 epoch 结束报告前刷新进度条
            self.state.curr_epoch += 1
            self.__trigger_events(Events.EPOCH_COMPLETED)

            # TODO: [CRITICAL] align the protocol
            # TODO: [关键] 对齐协议
            self.state.run_accumulated_output.append(
                self.state.epoch_accumulated_output
            )

        return


