import matplotlib.pyplot as plt

def compare_compressed(orig, comp):
    plt.subplot(1, 2, 1)
    plt.imshow(orig, cmap='gray')  # 'gray' colormap for grayscale images

    plt.subplot(1, 2, 2)
    plt.imshow(comp, cmap='gray')

    plt.title('Original vs. Compressed')
    plt.colorbar()  # Add a colorbar for reference
    plt.show()

#def scree_plot()