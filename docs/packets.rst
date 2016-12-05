Supported Packets
=================

Only some of the types of GRIPS packets are currently supported.

========  ========  ======  ===========
Support?  SystemID  TmType  Description
========  ========  ======  ===========
\         0x00      0x02    Flight software data rates
\         0x00      0x03    Flight software telemetry settings
\         0x00      0x0E    Flight software error
\         0x00      0x0F    Flight software message
\         0x01      0x02    Flight software watchdog
\         0x02      0x02    Flight software watchpup
partial   0x03      0x02    To pointing control system computer (PCSC)
partial   0x03      0x03    From pointing control system computer (PCSC)
full      0x05      0x02    GPS
\         0x07      0x02    Temperatures, through flight computer
\         0x07      0x03    Thermostats
\         0x0A      0x02    PDU voltages and currents
\         0x0A      0x03    Cryostat temperatures
\         0x0A      0x04    Power statuses
\         0x0C      0x02    Cryocooler housekeeping
\         0x0D      0x02    Not-dead-yet information
\         0x0F      0x02    MPPT housekeeping
\         0x10      0x1?    Ge quicklook spectrum and rates
\         0x10      0x20    Ge quicklook hitmap strip
\         0x40      0x02    Aspect housekeeping
\         0x40      0x03    Aspect settings
\         0x40      0x04    Temperatures, through aspect computer
\         0x40      0x10    Aspect science
\         0x4A      0x20    Aspect pitch-yaw image row
\         0x4B      0x20    Aspect roll image row
\         0x8?      0x02    Card cage (Ge) housekeeping
\         0x8?      0x04    Card cage (Ge) ASIC registers, LV top
\         0x8?      0x05    Card cage (Ge) ASIC registers, LV bottom
\         0x8?      0x06    Card cage (Ge) ASIC registers, HV top
\         0x8?      0x07    Card cage (Ge) ASIC registers, HV bottom
partial   0x8?      0x08    Card cage (Ge) counters
\         0x8?      0x09    Card cage (Ge) temperatures
full      0x8?      0xF1    Card cage (Ge) event, raw format, LV side
full      0x8?      0xF2    Card cage (Ge) event, raw format, HV side
full      0x8?      0xF3    Card cage (Ge) event, normal format
\         0x8?      0xFC    Card cage (Ge) channel enables
\         0x9?      0x04    Card cage (Ge) ASIC register differences, LV top
\         0x9?      0x05    Card cage (Ge) ASIC register differences, LV bottom
\         0x9?      0x06    Card cage (Ge) ASIC register differences, HV top
\         0x9?      0x07    Card cage (Ge) ASIC register differences, HV bottom
\         0xB6      0x02    Shield (BGO) housekeeping
\         0xB6      0x09    Shield (BGO) temperatures
\         0xB6      0x13    Shield (BGO) configuration
full      0xB6      0x80    Shield (BGO) events, 8 bytes per event
full      0xB6      0x81    Shield (BGO) counter rates
full      0xB6      0x82    Shield (BGO) events, 5 bytes per event
\         0xC0      0x02    SMASH housekeeping
\         0xC0      0x03    SMASH log messages
\         0xC0      0x10    SMASH science
========  ========  ======  ===========
