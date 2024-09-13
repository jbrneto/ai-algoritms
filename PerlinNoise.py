from perlin_noise import PerlinNoise

def perlin_img_gen(img_shape=(256,256,3), batch_size=32, octaves=5):
    xs = []
    ys = []

    for _ in range(batch_size):
        noise = PerlinNoise(octaves=octaves)
        xpix, ypix = img_shape[0], img_shape[1]
        
        ns = np.array([[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)])
        
        # black and white normalization
        ns[ns < 0] = 0
        ns = ns/np.max(ns)
        ns = 1.0 - ns

        img = np.zeros(img_shape)
        for c in range(img_shape[2]): # same image in all channels
            img[:,:,c] = ns
            img[:,:,c] = ns
            img[:,:,c] = ns

        msk = 1.0 - img[:,:,:1]
        msk[msk > 0.1] = 1.0

        xs.append(img)
        ys.append(msk)
    
    return xs, ys

# Usage
xs, ys = perlin_img_gen(batch_size=32)