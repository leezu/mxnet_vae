"""MNIST VAE.

Usage:
  main.py train [options]
  main.py continue [options]
  main.py test [options]
  main.py (-h | --help)

Options:
  -h --help                   Show this screen.
  -b --batch_size=<b>         Batch size [default: 128].
  -l --learning_rate=<lr>     Batch size [default: 5E-4].
  -L --lambda_l2_reg=<lr>     Lambda L2 regularization [default: 1E-5].
  -e --max_epochs=<e>         Number of epochs [default: 10].
  -g --gpu=<e>                Number of GPU [default: 0].
  -v --visualize              Show generated digits during training process
  --log=<log>                 Directory for logging [default: ./log/]

"""

import mxnet as mx
import vae
import data
import logging
import os.path
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt

IMG_DIM = 28
ARCHITECTURE = [IMG_DIM**2, 500, 500, 2]

HYPERPARAMS = {
    #"dropout": 0.9,
    "nonlinearity": "elu",
    "squashing": "squashing"
}

LOG_DIR = "./log"

def main(docopts):
    docopts["--batch_size"] = int(docopts["--batch_size"])
    docopts["--gpu"] = int(docopts["--gpu"])
    docopts["--lambda_l2_reg"] = float(docopts["--lambda_l2_reg"])
    docopts["--learning_rate"] = float(docopts["--learning_rate"])
    docopts["--max_epochs"] = int(docopts["--max_epochs"])

    # Logging
    logging.basicConfig(level=logging.INFO)

    #
    # Following http://nbviewer.jupyter.org/github/dmlc/mxnet/blob/master/example/notebooks/simple_bind.ipynb
    #

    X, Y = data.get_mnist()
    iter = mx.io.NDArrayIter(data=X, label=Y, batch_size=docopts["--batch_size"], shuffle=True)


    if docopts["train"] or docopts["continue"]:
        m = vae.VAE(ARCHITECTURE)
        sym = m.training_model()

        dbatch = iter.next()
        exe = sym.simple_bind(ctx=mx.gpu(docopts["--gpu"]), data = dbatch.data[0].shape)

        args = exe.arg_dict
        grads = exe.grad_dict
        outputs = dict(zip(sym.list_outputs(), exe.outputs))

        if docopts["continue"]:
            loaded_args = mx.nd.load(os.path.join(docopts["--log"], "parameters"))
            for name in args:
                if name != "data":
                    args[name][:] = loaded_args[name]

        # Initialize parameters
        xavier = mx.init.Xavier()
        for name, nd_array in args.items():
            if name != "data":
                xavier(name, nd_array)

        optimizer = mx.optimizer.create(name="adam",
                                        learning_rate=docopts["--learning_rate"],
                                        wd=docopts["--lambda_l2_reg"])
        updater = mx.optimizer.get_updater(optimizer)

        # Train
        keys = sym.list_arguments()
        optimizer = mx.optimizer.Adam()

        if docopts["--visualize"]:
            # Random image
            last_image_time = time.time()
            plt.ion()
            figure = plt.figure()
            imshow = plt.imshow(np.random.uniform(size=(28,28)), cmap="gray")

        for epoch in range(docopts["--max_epochs"]):
            iter.reset()
            epoch_start_time = time.time()
            batch = 0
            for dbatch in iter:
                args["data"][:] = dbatch.data[0]

                exe.forward(is_train=True)
                exe.backward()

                if docopts["--visualize"]:
                    # Throttle refresh ratio
                    if time.time() - last_image_time > 0.1:
                        last_image_time = time.time()
                        imshow.set_data(exe.outputs[2][
                            random.randint(0, docopts["--batch_size"])].reshape(
                                (28,28)).asnumpy())
                        figure.canvas.draw()
                        figure.canvas.flush_events()

                for index, key in enumerate(keys):
                    updater(index=index, grad=grads[key], weight=args[key])

                kl_divergence = exe.outputs[3].asnumpy()
                cross_entropy = exe.outputs[4].asnumpy()

                logging.info("Batch %d: %f mean kl_divergence", batch, kl_divergence.mean())
                logging.info("Batch %d: %f mean cross_entropy", batch, cross_entropy.mean())

                batch += 1

            logging.info("Finish training epoch %d in %f seconds",
                         epoch,
                         time.time() - epoch_start_time)

        # Save model parameters (including data, to simplify loading / binding)
        mx.nd.save(os.path.join(docopts["--log"], "parameters"),
                   {x[0]: x[1] for x in args.items() if x[0] != "data"})

    elif docopts["test"]:
        from matplotlib.widgets import Button

        m = vae.VAE(ARCHITECTURE)
        sym = m.testing_model()

        exe = sym.simple_bind(ctx=mx.gpu(docopts["--gpu"]),
                              data=(docopts["--batch_size"], ARCHITECTURE[-1]))

        args = exe.arg_dict
        grads = exe.grad_dict
        outputs = dict(zip(sym.list_outputs(), exe.outputs))

        loaded_args = mx.nd.load(os.path.join(docopts["--log"], "parameters"))
        for name in args:
            if name != "data":
                args[name][:] = loaded_args[name]

        args["data"][:] = np.random.randn(docopts["--batch_size"], ARCHITECTURE[-1])
        exe.forward(is_train=True)
        # testing_model has only 1 output
        batch = exe.outputs[0].asnumpy().reshape(-1, 28, 28)
        np.save(os.path.join(docopts["--log"], "output"), batch)

        imshow = plt.imshow(batch[0], cmap="gray")
        callback = Index(imshow, batch)
        axnext = plt.axes([0.8, 0.7, 0.1, 0.075])
        axprev = plt.axes([0.8, 0.6, 0.1, 0.075])
        next_button = Button(axnext, 'Next')
        next_button.on_clicked(callback.next)
        prev_button = Button(axprev, 'Previous')
        prev_button.on_clicked(callback.prev)

        plt.show()
        plt.waitforbuttonpress()

class Index(object):
    def __init__(self, imshow, batch):
        self.imshow = imshow
        self.ind = 0
        self.batch = batch
        self.batch_size = len(batch)

    def next(self, event):
        self.ind = (self.ind + 1) % self.batch_size
        self.imshow.set_data(self.batch[self.ind])
        plt.draw()

    def prev(self, event):
        self.ind = (self.ind - 1) % self.batch_size
        self.imshow.set_data(self.batch[self.ind])
        plt.draw()

if __name__ == "__main__":
    docopts = docopt(__doc__)

    if not os.path.exists(docopts["--log"]):
        os.makedirs(docopts["--log"])

    main(docopts)
