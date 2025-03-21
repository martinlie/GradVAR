class EarlyStopping:
      def __init__(self, patience=5, min_delta=0.001):
            """
            Early stopping mechanism, with patience
            If the loss improves by at least min_delta, the counter resets.
            If no improvement is observed for patience consecutive epochs, training stops.

            Args:
            - patience (int): Number of epochs to wait after loss stops improving.
            - min_delta (float): Minimum change in loss to be considered an improvement.
            """
            self.patience = patience
            self.min_delta = min_delta
            self.best_loss = float('inf')
            self.counter = 0

      def __call__(self, loss):
            if loss < self.best_loss - self.min_delta:
                  self.best_loss = loss
                  self.counter = 0  # Reset counter if improvement is found
            else:
                  self.counter += 1  # Increase counter if no improvement
        
            return self.counter >= self.patience  # Stop training if patience exceeded