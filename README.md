# verse-nnenum-int
# Assured Collision Avoidance for Learned Controllers: A Case Study of ACAS Xu

This repository is built based on top of Verse's [offical repository]([https://github.com/googleinterns/IBRNet](https://github.com/AutoVerse-ai/Verse-library))


## Introduction

This paper introduces a novel approach to verification of neural network controlled systems, combining the capabilities of the nnenum framework with the Verse toolkit. Addressing a critical gap in the traditional verification process which often does not include the system dynamics analysis while computing the neural network outputs, our integrated methodology enhances the precision and safety of decision-making in complex dynamical systems. By iteratively verifying neural network decisions and propagating system states, we maintain an accurate representation of the system’s behavior over time, a vital aspect in ensuring operational safety.
Our approach is exemplified through the verification of the neural network controlled Airborne Collision Avoidance System for Unmanned Aircraft (ACAS Xu). We demonstrate that the integration of nnenum and Verse not only accurately computes reachable sets for the UAS but also effectively handles the inherent complexity and nonlinearity of the system. The resulting analysis provides a nuanced understanding of the system’s behavior under varying operational conditions and interactions with other agents, such as intruder aircraft. The comprehensive simulations conducted as part of this study reveal the robustness of our approach, validating its effectiveness in verifying the safety and reliability of learned controllers. Furthermore, the scalability and adaptability of our methodology suggest its broader applicability in various autonomous systems requiring rigorous safety verification.

## Installation

Clone this repository:

```bash
https://github.com/gokulp01/verse-nnenum-int
verse-nnenum-int
```

The code is tested with python 3.9, cuda == 11.1, pytorch == 1.10.1. Additionally dependencies include: 

```bash
ray~=2.4.0
astunparse~=1.6.3
beautifulsoup4~=4.11.1
intervaltree~=3.1.0
lxml~=4.9.1
matplotlib~=3.4.3
numpy~=1.24
plotly~=5.8.2
polytope~=0.2.3
Pympler~=1.0.1
pyvista~=0.35.2
scipy~=1.9
six~=1.14.0
sympy~=1.6.2
torch~=1.13.1
tqdm~=4.64.1
z3-solver~=4.8.17.0
treelib~=1.6.1
portion~=2.3.1
graphviz~=0.20
networkx~=2.8.3
```

## Datasets

We reuse the ACAS Xu dataset from Stanley Bak's [ACASXu Falsification Benchmark(https://github.com/stanleybak/acasxu_closed_loop_sim/tree/main). Additionally, we also use the vnnlib and onnx files that can be found [here](https://github.com/stanleybak/nnenum/blob/master/examples/acasxu/data/ACASXU_run2a_3_2_batch_2000.onnx). 

For ease, we consolidate the instructions below:
```bash
mkdir data
cd data/
git clone https://github.com/stanleybak/nnenum.git
cp-r nnenum/examples/acasxu .
```
<div align="center">
<img width="423" alt="image" src="https://github.com/gokulp01/verse-nnenum-int/assets/43350089/53625011-1346-4a6c-8851-efeb541b3e9c">
</div>


## Usage
Use the nnenum scripts provided within this repository to generate verified trajectories of the agent (decision generation -- Section V-A). 
```bash
python3 -m nnenum.nnenum examples/acasxu/data/ACASXU_run2a_3_3_batch_2000.onnx examples/acasxu/data/prop_9.vnnlib
```
This should generate two `.npy` files. 

<div align="center">
<img width="382" alt="image" src="https://github.com/gokulp01/verse-nnenum-int/assets/43350089/14d786fb-af33-4b19-853e-04dd454f8297">
</div>

Now run the integration file: 
```bash
python3 acas_nnenum_integration.py
```
This integrates with Verse and generates the reachtubes. 

<div align="center">
<img width="376" alt="image" src="https://github.com/gokulp01/verse-nnenum-int/assets/43350089/59cd9581-013c-4f50-9996-ba007f71fad8">
</div>

## Cite this work

If you find our work / code implementation useful for your own research, please cite our paper.

```
@inproceedings{puthumanaillam2024assured,
  title={Assured Collision Avoidance for Learned Controllers: A Case Study of ACAS Xu},
  author={Puthumanaillam, Gokul and Vora, Manav Ketan and Shafa, Taha and Li, Yangge and Ornik, Melkior and Mitra, Sayan},
  booktitle={AIAA SCITECH 2024 Forum},
  pages={1168},
  year={2024}
}
}
```
