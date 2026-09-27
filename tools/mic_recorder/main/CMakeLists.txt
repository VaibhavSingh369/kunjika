# IDF 5.3 or newer. On IDF 5.0-5.2 replace the PRIV_REQUIRES line with:
#   PRIV_REQUIRES driver
idf_component_register(SRCS "main.c"
                       INCLUDE_DIRS "."
                       PRIV_REQUIRES esp_driver_i2s esp_driver_uart)
