package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "raytracer",
    srcs = [
        "main.cpp",
    ],
    deps = [
        ":lib",
    ],
)

cc_library(
    name = "lib",
    srcs = ["RayTracer.cpp"],
    hdrs = ["RayTracer.h"],
    deps = [
        ":bsp",
        ":camera",
        ":image",
        ":light",
        ":vector",
        "//raytracer/materials",
        "//raytracer/materials:normal_map",
        "//raytracer/objects",
    ],
)


cc_library(
    name = "boundaries",
    srcs = ["Boundaries.cpp"],
    hdrs = ["Boundaries.h"],
    deps = [
        ":ray",
        ":vector",
    ],
)

cc_library(
    name = "bsp",
    srcs = ["BSP.cpp"],
    hdrs = ["BSP.h"],
    deps = [
        ":boundaries",
        ":intersection",
        "//raytracer/objects:object",
    ],
)

cc_library(
    name = "camera",
    srcs = ["Camera.cpp"],
    hdrs = ["Camera.h"],
    deps = [
        ":vector",
    ],
)

cc_library(
    name = "color",
    srcs = ["Color.cpp"],
    hdrs = ["Color.h"],
)

cc_library(
    name = "image",
    srcs = ["Image.cpp"],
    hdrs = ["Image.h"],
    deps = [
        ":color",
    ],
)

cc_library(
    name = "intersection",
    srcs = ["Intersection.cpp"],
    hdrs = ["Intersection.h"],
    deps = [
        ":color",
        ":ray",
        ":vector",
        "//raytracer/objects:object",
        "//raytracer/materials:material",
    ],
)

cc_library(
    name = "light",
    srcs = ["Light.cpp"],
    hdrs = ["Light.h"],
    deps = [
        ":vector",
    ],
)

cc_library(
    name = "ray",
    srcs = ["Ray.cpp"],
    hdrs = ["Ray.h"],
    deps = [
        ":vector",
    ],
)

cc_library(
    name = "vector",
    srcs = ["Vector.cpp"],
    hdrs = ["Vector.h"],
)
