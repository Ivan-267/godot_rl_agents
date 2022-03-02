extends ISensor3D
class_name RayCastSensor3D
tool

export(float,2, 16,2) var n_rays_width := 6.0 setget set_n_rays_width
export(float,2, 16,2) var n_rays_height := 6.0 setget set_n_rays_height
export(float,1.0,100,0.5) var ray_length := 10.0 setget set_ray_length
export(float,10,360,10.0) var cone_width := 60.0 setget set_cone_width
export(float,10,180,10.0) var cone_height := 60.0 setget set_cone_height

var rays := []
var geo = null

func set_ray_length(value):
    ray_length = value
    if Engine.editor_hint:
        _spawn_nodes()
        
func set_n_rays_width(value):
    n_rays_width = value
    if Engine.editor_hint:
        _spawn_nodes()
        
func set_cone_width(value):
    cone_width = value
    if Engine.editor_hint:
        _spawn_nodes()
        
func set_n_rays_height(value):
    n_rays_height = value
    if Engine.editor_hint:
        _spawn_nodes()
        
func set_cone_height(value):
    cone_height = value
    if Engine.editor_hint:
        _spawn_nodes()


func _ready() -> void:
   _spawn_nodes()


func _spawn_nodes():
    print("spawning nodes")
    for ray in rays:
        ray.queue_free()
    if geo:
        geo.clear()
    #$Lines.remove_points()
    rays = []
    
    var horizontal_step = cone_width / (n_rays_width)
    var vertical_step = cone_height / (n_rays_height)
    
    var horizontal_start = horizontal_step/2 - cone_width/2
    var vertical_start = vertical_step/2 - cone_height/2   
    
    
    var points = []
    
    for i in n_rays_width:
        for j in n_rays_height:
            var angle_w = horizontal_start + i * horizontal_step
            var angle_h = vertical_start + j * vertical_step
            #angle_h = 0.0
            var ray = RayCast.new()
            var cast_to = to_spherical_coords(ray_length, angle_w, angle_h)
#            var cast_to = Vector3(
#                ray_length * sin(deg2rad(angle_w)),
#                ray_length * sin(deg2rad(angle_h)),
#                ray_length*cos(deg2rad(angle_w))*cos(deg2rad(angle_h))
#                #ray_length*sin(deg2rad(angle_w)) + ray_length*sin(deg2rad(angle_h))
#            )  
            ray.set_cast_to(cast_to)
            points.append(cast_to)
            
            ray.set_name("node_"+str(i)+" "+str(j))
            ray.enabled  = true
            ray.collide_with_areas = true
            add_child(ray)
            rays.append(ray)
            ray.force_raycast_update()
#            if Engine.editor_hint:
#                geo = ImmediateGeometry.new()
#
#
#
#                $Lines.add_point(
#                    Vector3.ZERO,
#                    cast_to
#                )
            
    if Engine.editor_hint:
        _create_debug_lines(points)
        
func _create_debug_lines(points):
    if not geo: 
        geo = ImmediateGeometry.new()
        add_child(geo)
        
    geo.clear()
    geo.begin(Mesh.PRIMITIVE_LINES)
    for point in points:
        geo.set_color(Color.aqua)
        geo.add_vertex(Vector3.ZERO)
        geo.add_vertex(point)
    geo.end()
    
    var mat = geo.get_surface_material(0)
    mat.albedo_color = Color(randf(), randf(), randf())
    

func display():
    if geo:
        geo.display()
        
    
    
func to_spherical_coords(r, inc, azimuth) -> Vector3:
    return Vector3(
        r*sin(deg2rad(inc))*cos(deg2rad(azimuth)),
        r*sin(deg2rad(azimuth)),
        r*cos(deg2rad(inc))*cos(deg2rad(azimuth))       
       )