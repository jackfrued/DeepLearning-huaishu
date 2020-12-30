from tensorboardX import SummaryWriter

class SummaryTool:
    def __init__(self, output_dir, record_step=5):
        self.record_step = record_step
        self.writer = SummaryWriter(logdir = output_dir)

    def add_scalar(self, scalar_name, scalar_value, global_step = None, walltime = None):
        self.writer.add_scalar(scalar_name, scalar_value, global_step)

    def add_scalars(self, scalar_name, scalar_value, global_step = None, walltime = None):
        self.writer.add_scalars(scalar_name, scalar_value, global_step)

    def close_tool(self):
        self.writer.close()
