EXTRA_CFLAGS += -DCONFIG _LITTLE_ENDIAN
EXTRA_CFLAGS += -DCONFIG_IOCTL_CFG80211 -DRTW_USE_CFG80211_STA_EVENT
ARCH ?= arm64
CROSS COMPILE ?=
KVER ?= $(shell uname-r)
ksrc:=lib/modules/$(KVER)/build 
MODDESTDIR := /lib/modules/$(KVER)/kernel/drivers/net/wireless/ 
INSTALL PREFIX :=


###Notify SDIO Host Keep Power During Syspend### 
CONFIG_RTW_SDIO_PM_KEEP_POWER =y
###MP HW TX MODE FOR VHT###
CONFIG_MP_VHT_HW_TX_MODE=n
###Platform Related###
CONFIG_PLATFORM_I386 PC =n 
CONFIG_PLATFORM_ANDROID_X86=n 
CONFIG_PLATFORM_ANDROID_INTEL_X86=n 
CONFIG_PLATFORM_=n 
CONFIG_PLATFORM_ARM_S3C2K4=n 
CONFIG_PLATFORM_ARM_PXA2XX =n 
CONFIG_PLATFORM_ARM_S3C6K4 =y 
CONFIG PLATFORM MIPS RMI =n 
CONFIG PLATFORMrtD2880B=n 
CONFIG_PLATFORMMIPS_AR9132=n 
CONFIG PLATFORM_RTK DMP =n 
CONFIG_PLATFORM_MIPS_PLM =n 
CONFIG_ PLATFORM MSTAR3389=n 
CONFIG_PLATFORM_MT53XX =n 
CONFIG_PLATFORM_ARM MX51_241H =n 
CONFIG_PLATFORM_FS_MX61=n 
--VISUAL--

ctrl_interface=DIR=/var/run/wpa_supplicant
update_config=1
country=CN

network={
        ssid="xxx-xxx"
        psk="xxx"
        key_mgmt=WPA-PSK
        disabled=1
}

network={
        ssid="xxx-xxx"
        psk="xxx"
        key_mgmt=WPA-PSK
}
