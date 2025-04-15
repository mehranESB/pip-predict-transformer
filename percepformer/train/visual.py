import matplotlib.pyplot as plt
import json


class TrainingPlotter:
    def __init__(self):
        """
        Initializes the plotter.
        """
        self.train_losses = []  # Store training loss values for plotting
        self.val_losses = []  # Store validation loss values for plotting
        self.train_iter = []  # Store iteration numbers for training
        self.valid_iter = []  # Store iteration numbers for validation
        self.fig, self.ax = plt.subplots()  # Create a figure and axis for plotting

    def update(self, iteration, train_loss=None, val_loss=None):
        """
        Updates the plot with the current training loss and optional validation loss.

        Parameters:
            iteration (int): The current iteration number.
            train_loss (float): The training loss at this iteration.
            val_loss (float, optional): The validation loss at this epoch (default is None).
        """
        # Append the current iteration and training loss values
        if train_loss is not None:
            self.train_iter.append(iteration)
            self.train_losses.append(train_loss)

        # If it's the end of an epoch, append the validation loss
        if val_loss is not None:
            self.valid_iter.append(iteration)
            self.val_losses.append(val_loss)

        # Clear the previous plot lines but keep the axis
        self.ax.clear()

        # Plot training loss
        self.ax.plot(
            self.train_iter, self.train_losses, label="Training Loss", color="blue"
        )

        # Plot validation loss (only after each epoch)
        self.ax.plot(
            self.valid_iter,
            self.val_losses,
            label="Validation Loss",
            color="red",
            linestyle="--",
        )

        # Customize plot appearance
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training and Validation Loss")
        self.ax.legend()
        self.ax.grid(True)
        self.fig.tight_layout()

        # Pause to update the plot in real-time
        plt.pause(0.05)

    def show(self):
        """
        Displays the final plot.
        """
        plt.show()

    def save_plot(self, filename="training_plot.png", json_save=False):
        """
        Save the plot as an image file, and optionally save the losses as a JSON file.

        Parameters:
            filename (str): The name of the file to save the plot as (default is "training_plot.png").
            json_save (bool): Whether to save the losses as a JSON file (default is False).
        """
        # Save the plot as an image
        self.fig.savefig(filename)

        # Save losses as JSON file if needed
        if json_save:
            data = {
                "train_iter": self.train_iter,
                "train_losses": self.train_losses,
                "valid_iter": self.valid_iter,
                "val_losses": self.val_losses,
            }

            # Determine the name for the JSON file based on the plot filename
            json_filename = str(filename).replace(".png", "_losses.json")

            # Save the losses dictionary to the JSON file
            with open(json_filename, "w") as f:
                json.dump(data, f, indent=4)

            print(f"Losses saved to {json_filename}")

    def close_fig(self):
        plt.close(self.fig)
