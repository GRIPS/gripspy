Quick Start
===================================

Simply import `gripspy` to start.

.. code-block:: python

   import gripspy

The classes for handling science data are under `gripspy.science`.
The classes for handling housekeeping data are under `gripspy.housekeeping`.
Here are some examples of what has been implemented so far:

.. code-block:: python

   data = gripspy.science.GeData(detector_number, filename)
   data = gripspy.science.BGOCounterData(filename)
   data = gripspy.science.BGOEventData(filename)
   data = gripspy.science.PYSequence(filelist)
   data = gripspy.housekeeping.GPSData(filename)
