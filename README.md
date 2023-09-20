DC Gan with automatic mixed precision, the system uses BCEWithLogitsLoss, so we need to remove the sigmoid layer that's at the end of the Discriminator. 
