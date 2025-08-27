---
hide:
 - toc
---

OpenDSM's origin lies within the CalTRACK methodology. 

The CalTRACK methodology comes from a working group initiated by California Public Utility Commission (CPUC) in 2012 to track metered savings for pay-for-performance energy efficiency programs. 

The first versions were developed collaboratively by PG&E and other California IOU's in conjunction with the California Energy Commission (CEC), Recurve, and other stakeholders. The goal was to develop reliable and transparent processes to calculate avoided energy use. The models that came from this process were the daily/billing models and the hourly model. The daily/billing models were heavily influenced by the Princeton Scorekeeping Method <a href="https://marean.mycpanel.princeton.edu/images/prism_intro.pdf">(PRISM 1986) [1]</a>. Similarly the hourly model was an adapted version of Lawrence Berkeley National Laboratory's Time-of-Week and Temperature (TOWT) model <a href="https://eta-publications.lbl.gov/sites/default/files/LBNL-4944E.pdf">(TOWT 2011) [2]</a>.

The CalTRACK methodology only defined how to set up the models; it was not itself a library that users could utilize to make measurements. OpenEEmeter came in to become the open-source Python implementation of the CalTRACK methods. The changes to the models since the CalTRACK methods can be found in each model's references section. 

OpenEEmeter has since become OpenDSM to be able to include comparison groups and `EEweather`.

Today, OpenDSM has moved beyond the CalTRACK methods. Our models are more accurate, proven in extensive testing, easier to use, and significantly faster. OpenDSM continues to evolve and develop, but all of our efforts are the result of the foundational work from those before us.


[1] PRISM 1986: <a href="https://doi.org/10.1016/0378-7788(86)90003-4">https://doi.org/10.1016/0378-7788(86)90003-4</a>

[2] TOWT 2011: <a href="https://doi.org/10.1109/TSG.2011.2145010">https://doi.org/10.1109/TSG.2011.2145010</a>