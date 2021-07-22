# Note

This is an old approach that's now abandoned. I'm keeping this here for referential purposes. Here, I tried replicating ElBadawy's work from scratch. I couldn't get signifcantly great results namely cause the actual model uses Instance Normalisation and not Batch Normalisation layers (as reported), and also I should have followed the UNIT codebase more closely from the start (I initially followed a CycleGAN implementation which lead to a confused design). 

This approach was a gross amulgumation of CycleGAN and UNIT. Though they have similarities, especially in the cyclic reconstruction, I was messing up how the basic VAE reconstruction was implemented. 

Anyways later, ElBadawy released the code which I've switched to for effectiveness as well as cleanliness. I've also been involved in contributing to it by adding some features from this abandoned codebase to it, as well as finishing some parts of the pipeline that weren't included. Refer to the main branch for a fork to the VAE-GAN repo titled `voice_conversion`
