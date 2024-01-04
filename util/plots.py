import matplotlib.pyplot as plt

def compare_compressed(orig, comp):
    plt.subplot(1, 2, 1)
    plt.imshow(orig, cmap='gray')  # 'gray' colormap for grayscale images

    plt.subplot(1, 2, 2)
    plt.imshow(comp, cmap='gray')

    plt.title('Original vs. Compressed')
    plt.colorbar()  # Add a colorbar for reference
    plt.show()

def scree_plot(S):
    plt.plot(np.arange(1, len(S) + 1), S, marker='o')
    plt.title('Scree Plot')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value')
    plt.show()
