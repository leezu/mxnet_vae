import mxnet as mx
import vae
import data
import logging
from opterator import opterate

@opterate
def main(action,
         parameters="./parameters",
         gpu=0,
         batch_size=128,
         learning_rate=0.000001,
         num_epochs=1000):
    gpu = int(gpu)
    batch_size = int(batch_size)
    learning_rate = float(learning_rate)
    num_epochs = int(num_epochs)

    network_structure = [784, 500, 250, 10]

    # Logging
    logging.basicConfig(level=logging.INFO)

    #
    # Following http://nbviewer.jupyter.org/github/dmlc/mxnet/blob/master/example/notebooks/simple_bind.ipynb
    #

    X, Y = data.get_mnist()
    iter = mx.io.NDArrayIter(data=X, label=Y, batch_size=batch_size, shuffle=True)


    if action == "train" or action == "continue":
        m = vae.VAE(network_structure)
        sym = m.training_model()


        if action == "train":
            dbatch = iter.next()
            exe = sym.simple_bind(ctx=mx.gpu(gpu), data = dbatch.data[0].shape)
        elif action == "continue":
            exe = sym.bind(ctx=mx.gpu(gpu), args = mx.nd.load(parameters))

        args = exe.arg_dict
        grads = exe.grad_dict
        outputs = dict(zip(sym.list_outputs(), exe.outputs))

        # Initialize parameters
        xavier = mx.init.Xavier()
        for name, nd_array in args.items():
            if name != "data":
                xavier(name, nd_array)

        # Optimizer
        def SGD(key, weight, grad, lr=0.1, grad_norm=batch_size):
            # key is key for weight, we can customize update rule
            # weight is weight array
            # grad is grad array
            # lr is learning rate
            # grad_norm is scalar to norm gradient, usually it is batch_size
            norm = 1.0 / grad_norm
            # here we can bias' learning rate 2 times larger than weight
            if "weight" in key or "gamma" in key:
                weight[:] -= lr * (grad * norm)
            elif "bias" in key or "beta" in key:
                weight[:] -= 2.0 * lr * (grad * norm)
            else:
                pass

        # Train
        keys = sym.list_arguments()
        optimizer = mx.optimizer.Adam()

        for epoch in range(num_epochs):
            iter.reset()
            batch = 0
            for dbatch in iter:
                args["data"][:] = dbatch.data[0]

                exe.forward(is_train=True)
                exe.backward()

                for key in keys:
                    SGD(key, args[key], grads[key], lr=learning_rate)

                kl_divergence = exe.outputs[3].asnumpy()
                cross_entropy = exe.outputs[4].asnumpy()

                logging.info("Batch %d: %f mean kl_divergence", batch, kl_divergence.mean())
                logging.info("Batch %d: %f mean cross_entropy", batch, cross_entropy.mean())

                batch += 1

            logging.info("Finish training epoch %d", epoch)

        # Save model parameters
        mx.nd.save(parameters, args)

    elif action == "test":
        m = vae.VAE(network_structure)
        sym = m.testing_model()
        exe = sym.bind(ctx=mx.gpu(gpu), args = mx.nd.load(parameters))

        args = exe.arg_dict
        outputs = dict(zip(sym.list_outputs(), exe.outputs))

        args["data"][:] = np.random.randn(batch_size, network_structure[-1])

        exe.forward(is_train=True)

        np.save("./output", exe.outputs[0].asnumpy())


if __name__ == "__main__":
    main()
