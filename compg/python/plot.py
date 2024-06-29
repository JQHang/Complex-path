import matplotlib.pyplot as plt

def plot_histogram(histogram_data, file_path=None):
    """
    Plots a histogram from the given data and saves the plot to a file if a file path is provided.

    Parameters:
        histogram_data (list of tuples): List containing tuples of (bin_start, bin_end, count).
        file_path (str, optional): Path where the histogram image should be saved. If None, the plot is not saved.

    Returns:
        None
    """
    # Extract data for plotting
    bin_edges = [x[0] for x in histogram_data] + [histogram_data[-1][1]]
    counts = [x[2] for x in histogram_data]

    # Create histogram plot
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], counts, width=[bin_edges[i+1] - bin_edges[i] for i in range(len(bin_edges)-1)], align='edge', edgecolor='black')
    plt.xlabel('Value Range')
    plt.ylabel('Count')
    plt.title('Histogram of Data Bins')
    
    # Check if a file path is provided and save the plot
    if file_path:
        plt.savefig(file_path)
    
    # Display the plot
    plt.show()