# No REQUIRES list on purpose: main then sees every component in the build,
# including Wi-Fi/lwIP and the managed components from idf_component.yml.
idf_component_register(SRCS "main.cc" "kws_engine.cc" "net.c" "stream.c" "adpcm.c"
                       INCLUDE_DIRS ".")
