# Author: "SigTerm"
# From: https://blender.stackexchange.com/questions/18645/
# is-it-possible-to-render-an-armature

bl_info = {"name": "Armature To Mesh", "category": "Object"}

import bpy
import mathutils

class ArmatureToMesh(bpy.types.Operator):
    """Armature to mesh/skin conversion script"""
    bl_idname = "object.armature_to_mesh_skin"
    bl_label = "Armature to Mesh/Skin"
    bl_options = {'REGISTER', 'UNDO'}

    def decomposeMatrix(self, matrix):
        return [x.to_3d() for x in matrix.transposed()]

    def processArmature(self, context, arm, genVertexGroups = True):
        print("processing armature {0}".format(arm.name))

        if genVertexGroups:
            #because setting pose_position ot 'REST' manually doesn't work for some reason.
            genVertexGroups = arm.data.pose_position == 'REST'

        meshName = arm.name + "_mesh"
        meshData = bpy.data.meshes.new(meshName + "Data")
        meshObj = bpy.data.objects.new(meshName, meshData)
        meshObj.location = arm.location

        scene = context.scene
        #scene.objects.link(meshObj)
        # update 
        scene.collection.objects.link(meshObj)

        armMatrix = arm.matrix_local.copy()

        verts = []
        edges = []
        faces = []
        vertexGroups = {}


        for bone in arm.pose.bones:
            poseBone = bone
            boneName = bone.name
            armBone = arm.data.bones[boneName]
            #print(poseBone.matrix)
            #print(armBone.matrix)
            #print(boneName)

            boneMatrix = poseBone.matrix
            boneStart = poseBone.head
            boneEnd = poseBone.tail

            decomposedMatrix = self.decomposeMatrix(boneMatrix)
            #print(decomposedMatrix)
            xSize = armBone.bbone_x
            zSize = armBone.bbone_z
            #print(xSize, zSize)
            xSizeAdd = bone.x_axis
            zSizeAdd = bone.z_axis
            xSizeAdd = decomposedMatrix[0]
            zSizeAdd = decomposedMatrix[2]
            ySizeAdd = decomposedMatrix[1]
            origin = mathutils.Vector((0.0, 0.0, 0.0)) @ boneMatrix
            xSizeAdd *= xSize
            zSizeAdd *= zSize
            ySizeAdd *= bone.length

            baseIndex = len(verts)

            verts.append((boneStart - xSizeAdd + zSizeAdd)@armMatrix)
            verts.append((boneStart + xSizeAdd + zSizeAdd)@armMatrix)
            verts.append((boneStart - xSizeAdd - zSizeAdd)@armMatrix)
            verts.append((boneStart + xSizeAdd - zSizeAdd)@armMatrix)
            verts.append((boneEnd - xSizeAdd + zSizeAdd)@armMatrix)
            verts.append((boneEnd + xSizeAdd + zSizeAdd)@armMatrix)
            verts.append((boneEnd - xSizeAdd - zSizeAdd)@armMatrix)
            verts.append((boneEnd + xSizeAdd - zSizeAdd)@armMatrix)

            base = baseIndex
            newFaces = [
                (base+0, base+1, base+3, base+2),
                (base+5, base+4, base+6, base+7),
                (base+1, base+0, base+4, base+5),
                (base+2, base+3, base+7, base+6),
                (base+3, base+1, base+5, base+7),
                (base+0, base+2, base+6, base+4)
                ]
            faces.extend(newFaces)

            if genVertexGroups:
                vertexGroups[boneName] = [(x, 1.0) for x in range(baseIndex, len(verts))]

        meshData.from_pydata(verts, edges, faces)

        if genVertexGroups:
            for name, vertexGroup in vertexGroups.items():
                groupObject = meshObj.vertex_groups.new(name)
                for (index, weight) in vertexGroup:
                    groupObject.add([index], weight, 'REPLACE')

            modifier = meshObj.modifiers.new('ArmatureMod', 'ARMATURE')
            modifier.object = arm
            modifier.use_bone_envelopes = False
            modifier.use_vertex_groups = True

        meshData.update()

        return meshObj

    def processObject(self, context, obj):
        if (obj == None):
            return False
        if (obj.type != "ARMATURE"):
            print ("invalid type {0} of object {1}: armature expected".format(obj.type, obj.name))
            return False
        self.processArmature(context, obj)
        return True

    def execute(self, context):
            scene = context.scene
            selected = context.selected_objects
            processedAnything = False
            if len(selected) > 0:
                print ("selected objects present, processing selection")
                for obj in selected:
                    processedAnything |= self.processObject(context, obj)
                pass
            else:
                print ("processing active object")
                obj = context.active_object
                processedAnything |= self.processObject(context, obj)

            if not processedAnything:
                print ("no objects processed")

            return {'FINISHED'}

def register():
    bpy.utils.register_class(ArmatureToMesh)

def unregister():
    bpy.utils.unregister_class(ArmatureToMesh)

if __name__ == "__main__": 
    register()