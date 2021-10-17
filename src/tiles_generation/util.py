import cv2
import shapely


def show_small_img(img, name='img', show=True):
    if not show:
        return

    small_height = 512
    shape = img.shape[:2]
    img_small = cv2.resize(img, (small_height, int(small_height / shape[1] * shape[0])))
    cv2.imshow(name, img_small)
    cv2.waitKey(1)


def _convert_cv_contours_to_shapely(shapely_polygons, cv_contours, hierarchy, current_contour_index, is_hole, current_holes):
    if current_contour_index == -1:
        return

    while True:
        contour = cv_contours[current_contour_index]
        if len(contour) >= 3:
            first_child = hierarchy[current_contour_index][2]
            internal_holes = []
            _convert_cv_contours_to_shapely(
                shapely_polygons=shapely_polygons,
                cv_contours=cv_contours,
                hierarchy=hierarchy,
                current_contour_index=first_child,
                is_hole=not is_hole,
                current_holes=internal_holes)

            if is_hole:
                current_holes.append(contour)
            else:
                polygon = shapely.geometry.Polygon(contour, holes=internal_holes)
                shapely_polygons.append(polygon)

        current_contour_index = hierarchy[current_contour_index][0]
        if current_contour_index == -1:
            break


def convert_cv_contours_to_shapely(cv_contours, hierarchy):
    shapely_polygons = []
    _convert_cv_contours_to_shapely(
        shapely_polygons=shapely_polygons,
        cv_contours=cv_contours,
        hierarchy=hierarchy,
        current_contour_index=0,
        is_hole=False,
        current_holes=[])
    return shapely_polygons


def convert_cv_contours_to_shapely_geometry(cv_contours, hierarchy):
    polygons = convert_cv_contours_to_shapely(cv_contours, hierarchy[0])

    geometry = []
    for polygon in polygons:
        multipolygon = shapely.geometry.multipolygon.MultiPolygon([polygon])
        geometry.append(multipolygon)

    return geometry
