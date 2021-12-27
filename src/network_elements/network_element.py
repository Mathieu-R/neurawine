class NetworkElement:
  def __init__(self) -> None:
      raise NotImplementedError()
    
  def forward_propagate(self, *args, **kwargs):
    raise NotImplementedError()
  
  def backward_propagate(self, *args, **kwargs):
    raise NotImplementedError()
  
  def update_weights_and_bias(self, *args, **kwargs):
    pass