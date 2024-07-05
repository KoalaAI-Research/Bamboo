import matplotlib
matplotlib.use("agg")  # Set the backend to 'agg'
import matplotlib.pyplot as plt

# Create a basic plot
plt.plot([1, 2, 3], [4, 5, 6])

# Display the plot
#plt.show()

# Save the plot to a file
plt.savefig("lossTest.jpg")
