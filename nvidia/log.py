def log_progress(context, epoch, num_epochs, batch, num_samples, timemeter, lossmeter):
    """
    Prints current values and averages for AverageMeters
    
    Parameters:
    ----------
    context : str
        What progress we are tracking.
        Options - {'Train', 'Validation', 'Test'}
   
    epoch : int
        Current epoch.

    num_epochs : int
        Total number of epochs.

    batch : int
        batch number from Pytorch dataloader.

    num_samples : int
        number of minibatches. Given by `len(dataloader)`.

    timemeter : lungs.metrics.AverageMeter
        AverageMeter object recording runtime.

    lossmeter : lungs.metrics.AverageMeter
        AverageMeter object recording loss.
    """
    message = f"{context} Epoch: [{epoch}/{num_epochs}] "\
              f"Batch: [{batch}/{num_samples}] "\
              f"Time: {timemeter.val:.2f} [{timemeter.avg:.2f}] "\
              f"Loss: {lossmeter.val:.4f} [{lossmeter.avg:.4f}] "\

    logger.info(message)
