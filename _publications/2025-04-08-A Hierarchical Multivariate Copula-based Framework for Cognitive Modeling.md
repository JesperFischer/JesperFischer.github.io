---
title: "A Hierarchical Multivariate Copula-based Framework for Cognitive Modeling"
collection: publications
category: manuscripts
permalink: /publication/2025-04-08-A Hierarchical Multivariate Copula-based Framework for Cognitive Modeling
excerpt: 'Copula, Multivariate, Hierarchical; Framework'
date: 2025-04-08
venue: 'In Preparation'
githuburl: 'https://anonymous.4open.science/r/Hierarchical-Multivariate-Copula-Framework-D746/README.md'
paperurl: 'https://JesperFischer.github.io/files/CBM_8pager.pdf'
citation: ''
---

Computational cognitive models provide a principled approach to understanding behavior and cognition by 
formalizing latent parameters underlying decision-making and learning. 
Many existing models take a univariate approach, analyzing single measures in isolation, 
while others incorporate multiple measures but impose specific process assumptions that constrain 
how these measures relate. For example, drift diffusion models (DDMs) jointly model choices and response times under the assumption 
that decisions arise from an evidence accumulation process. While effective in many contexts, this assumption constrains the flexibility of DDMs, 
potentially limiting their applicability to cognitive processes that do not conform to an accumulation-to-bound mechanism.
Here, we introduce a hierarchical multivariate modeling framework that uses copulas to flexibly combine independent likelihood functions, 
enabling joint modeling of multiple measures without imposing restrictive assumptions. Through simulations and empirical applications, 
we assess the reliability, discriminability, and advantages of copula-based modeling (CBM). Model validation via simulation-based calibration, 
model recovery, and sensitivity analyses demonstrate that CBM is computationally robust and accurately recovers latent parameters. 
When applied to psychophysical and probabilistic learning tasks, CBM can be empirically distinguished from DDMs, even with limited data. 
Additionally, incorporating response times improves predictive accuracy for binary choices while reducing uncertainty in parameter estimates.
This framework offers several advantages. It accommodates diverse data types, including behavioral responses, physiological signals, 
and neural activity, without requiring them to share the same distributional properties. It explicitly models dependencies between measures, 
capturing interactions typically overlooked in univariate approaches. It is theoretically agnostic,
allowing for greater flexibility in modeling cognitive processes. 
Further, it enables more efficient use of available data by integrating multiple sources of information,
while enhancing model accuracy and efficiency of parameter estimation. These findings establish CBM as a flexible and generalizable framework,
broadening the scope of multivariate cognitive modeling and expanding the methodological toolkit available for cognitive science.