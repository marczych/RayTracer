package(default_visibility = ["//visibility:public"])

cc_library(
    name = "materials",
    deps = [
        ":air",
        ":checkerboard",
        ":criss_cross",
        ":flat_color",
        ":glass",
        ":marble",
        ":material",
        ":shiny_color",
        ":turbulence",
        ":wood",
    ],
)

cc_library(
    name = "material",
    srcs = ["Material.cpp"],
    hdrs = ["Material.h"],
    deps = [
        ":normal_map",
        "//raytracer:color",
    ],
)

cc_library(
    name = "normal_map",
    srcs = ["NormalMap.cpp"],
    hdrs = ["NormalMap.h"],
    deps = [
        ":perlin_noise",
        "//raytracer:vector",
    ],
)

cc_library(
    name = "perlin_noise",
    srcs = ["PerlinNoise.cpp"],
    hdrs = ["PerlinNoise.h"],
)


cc_library(
    name = "air",
    srcs = ["Air.cpp"],
    hdrs = ["Air.h"],
    deps = [
        ":material",
        "//raytracer:color",
        "//raytracer:vector",
    ],
)

cc_library(
    name = "criss_cross",
    srcs = ["CrissCross.cpp"],
    hdrs = ["CrissCross.h"],
    deps = [
        ":material",
        ":perlin_noise",
        "//raytracer:color",
        "//raytracer:vector",
    ],
)

cc_library(
    name = "checkerboard",
    srcs = ["Checkerboard.cpp"],
    hdrs = ["Checkerboard.h"],
    deps = [
        ":material",
        "//raytracer:color",
        "//raytracer:vector",
    ],
)

cc_library(
    name = "flat_color",
    srcs = ["FlatColor.cpp"],
    hdrs = ["FlatColor.h"],
    deps = [
        ":material",
        "//raytracer:color",
        "//raytracer:vector",
    ],
)

cc_library(
    name = "glass",
    srcs = ["Glass.cpp"],
    hdrs = ["Glass.h"],
    deps = [
        ":material",
        "//raytracer:color",
        "//raytracer:vector",
    ],
)

cc_library(
    name = "marble",
    srcs = ["Marble.cpp"],
    hdrs = ["Marble.h"],
    deps = [
        ":material",
        "//raytracer:color",
        "//raytracer:vector",
    ],
)

cc_library(
    name = "shiny_color",
    srcs = ["ShinyColor.cpp"],
    hdrs = ["ShinyColor.h"],
    deps = [
        ":material",
        "//raytracer:color",
        "//raytracer:vector",
    ],
)

cc_library(
    name = "turbulence",
    srcs = ["Turbulence.cpp"],
    hdrs = ["Turbulence.h"],
    deps = [
        ":material",
        "//raytracer:color",
        "//raytracer:vector",
    ],
)

cc_library(
    name = "wood",
    srcs = ["Wood.cpp"],
    hdrs = ["Wood.h"],
    deps = [
        ":material",
        "//raytracer:color",
        "//raytracer:vector",
    ],
)
