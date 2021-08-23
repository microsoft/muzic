# Muzic

Muzic is a research project that empowers music understanding and generation with deep learning and artificial intelligence. 
Muzic was started by [some researchers](https://www.microsoft.com/en-us/research/project/ai-music/) from [Microsoft Research Asia](https://www.microsoft.com/en-us/research/lab/microsoft-research-asia/).  


Our research work in Muzic:

* Music Score Generation
  + Song Writing: [SongMASS](https://arxiv.org/pdf/2012.05168.pdf)
  + Lyric Generation: [DeepRapper](https://arxiv.org/pdf/2107.01875.pdf)
  + Accompaniment Generation: [PopMAG](https://arxiv.org/pdf/2008.07703.pdf)
* Singing Voice Synthesis
  + [HiFiSinger](https://arxiv.org/pdf/2009.01776.pdf), [DeepSinger](https://arxiv.org/pdf/2007.04590.pdf), [XiaoiceSing](https://arxiv.org/pdf/2006.06261.pdf)
* Music Understanding
  + [MusicBERT](https://arxiv.org/pdf/2106.05630.pdf)


We initially release the code of two research work: [MusicBERT](musicbert) and [DeepRapper](deeprapper). You can find the README in the corresponding folder. 


## Requirements
The requirements for running muzic are listed in `requirements.txt`. To install the requirements, run:
```bash
pip install -r requirements.txt
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
