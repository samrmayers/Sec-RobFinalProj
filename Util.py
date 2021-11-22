# freezes a module so its weights aren't updated during the rest of the model training
def freeze_module(module):
    for layer in module.children():
        for param in layer.parameters():
            param.requires_grad = False


# displays an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()