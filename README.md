# MLE Pipeline - Predict image choices
This is more for creating a ML pipeline than for the model, however, the model is intended to take a stack of images
and predict whether the images would be picked by myself.

I made the [image-label-application](https://github.com/sora4222/image-label-application) to label the image dataset.

## Support requirements
Does not currently support MacOs as [tfx_bsl does not run on Apple Silicon chips](https://stackoverflow.com/questions/75611977/tensorflow-transform-installation-failure-on-mac-m2).
This does make it difficult for my current setup to use TFX for the pipeline.
A Windows computer does not support this either currently as `tensorflow-io-gcs-filesystem` does not support windows as
it is [missing a Windows build](https://discuss.tensorflow.org/t/tensorflow-io-gcs-filesystem-with-windows/18849/6).
To get around this you can use WSL to host an Ubuntu OS that can install everything appropriately.
