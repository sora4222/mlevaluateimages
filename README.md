# MLE Pipeline - Predict image choices
This is more for creating a ML pipeline than for the model, however, the model is intended to take a stack of images
and predict whether the images would be picked by myself.

I made the [image-label-application](https://github.com/sora4222/image-label-application) to label the image dataset.

## Support requirements
Does not currently support MacOs as [tfx_bsl does not run on Apple Silicon chips](https://stackoverflow.com/questions/75611977/tensorflow-transform-installation-failure-on-mac-m2).
This does make it difficult for my current setup to use TFX for the pipeline.
