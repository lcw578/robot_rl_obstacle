"""任务限位数学/XML测试；不加载MuJoCo、不运行机械臂或训练。"""
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from task_joint_limits import (apply_task_joint_limits,reference_limits_for_profile,
                               check_snapshot_limit_profile,update_position_actuator_ranges,
                               J16_360_PROFILE,J16_360_DEG)


class TaskJointLimitTests(unittest.TestCase):
    names=['shoulder_joint','upperArm_joint']

    def root(self,angle='radian'):
        root=ET.Element('mujoco');ET.SubElement(root,'compiler',angle=angle)
        body=ET.SubElement(ET.SubElement(root,'worldbody'),'body')
        for name in self.names:ET.SubElement(body,'joint',name=name,range='-3 3')
        return root

    def test_joint2_only_and_restore_reference_from_snapshot(self):
        root=self.root();reference=np.deg2rad([[-175,175],[-175,175]])
        effective=apply_task_joint_limits(root,self.names,reference,{'upperArm_joint':[-90,90]},[10.6,15.2])
        np.testing.assert_allclose(np.rad2deg(effective),[[-175,175],[-90,90]])
        np.testing.assert_allclose(np.fromstring(root.find(".//joint[@name='upperArm_joint']").get('range'),sep=' '),np.deg2rad([-90,90]))
        restored=apply_task_joint_limits(root,self.names,reference,{},[10.6,15.2])
        np.testing.assert_allclose(restored,reference)

    def test_degree_xml_uses_degrees(self):
        root=self.root('degree')
        apply_task_joint_limits(root,self.names,np.deg2rad([[-175,175]]*2),{'upperArm_joint':[-90,90]},[0,0])
        self.assertEqual(root.find(".//joint[@name='upperArm_joint']").get('range'),'-90 90')

    def test_invalid_config_and_home_rejected_before_xml_mutation(self):
        for limits,home in [({'typo':[-90,90]},[0,0]),
                ({'upperArm_joint':[-180,180]},[0,0]),
                ({'upperArm_joint':[90,-90]},[0,0]),
                ({'upperArm_joint':[-90,90]},[0,100]),
                ({'upperArm_joint':[float('nan'),90]},[0,0])]:
            root=self.root();before=ET.tostring(root)
            with self.subTest(limits=limits),self.assertRaises(ValueError):
                apply_task_joint_limits(root,self.names,np.deg2rad([[-175,175]]*2),limits,home)
            self.assertEqual(ET.tostring(root),before)

    def test_explicit_robot_profile_expands_only_j1_j6_and_writes_actual_xml(self):
        names=list(J16_360_DEG);root=ET.Element('mujoco');ET.SubElement(root,'compiler',angle='radian')
        body=ET.SubElement(ET.SubElement(root,'worldbody'),'body')
        for name in names:ET.SubElement(body,'joint',name=name,range='-3.0543 3.0543')
        actuators=ET.SubElement(root,'actuator')
        position=ET.SubElement(actuators,'position',joint='shoulder_joint',ctrlrange='-3.0543 3.0543')
        motor=ET.SubElement(actuators,'motor',joint='wrist3_joint',ctrlrange='-10 10')
        old=np.array([[-3.0543,3.0543]]*6)
        reference=reference_limits_for_profile(names,old,J16_360_PROFILE)
        expected=[[-360,360],[-175,175],[-175,175],[-175,175],[-175,175],[-360,360]]
        np.testing.assert_allclose(np.degrees(reference),expected)
        effective=apply_task_joint_limits(root,names,reference,{},[10.6,15.2,134.3,118.2,90,0])
        update_position_actuator_ranges(root,names,effective)
        np.testing.assert_allclose(np.fromstring(position.get('ctrlrange'),sep=' '),np.deg2rad([-360,360]))
        self.assertEqual(motor.get('ctrlrange'),'-10 10')
        for i,name in enumerate(names):
            np.testing.assert_allclose(np.fromstring(root.find(f".//joint[@name='{name}']").get('range'),sep=' '),effective[i])
        # 原先J1约174°再增加2°会越限；新型号下176°合法，J2仍不允许176°。
        self.assertTrue(effective[0,0]<np.deg2rad(176)<effective[0,1])
        self.assertGreater(np.deg2rad(176),effective[1,1])
        margins=np.degrees(np.minimum(np.deg2rad([173.9692,0,0,0,0,0])-reference[:,0],
                                      reference[:,1]-np.deg2rad([173.9692,0,0,0,0,0])))
        self.assertAlmostEqual(margins[0],186.0308)
        # 任务范围依然不能扩大选定的型号限位。
        with self.assertRaisesRegex(ValueError,'不能扩大'):
            apply_task_joint_limits(root,names,reference,{'shoulder_joint':[-361,361]},[0]*6)

    def test_profile_default_preserves_old_snapshot_and_explicit_mismatch_rejected(self):
        names=list(J16_360_DEG);old=np.array([[-3.0543,3.0543]]*6)
        np.testing.assert_array_equal(reference_limits_for_profile(names,old),old)
        check_snapshot_limit_profile({}, {})
        check_snapshot_limit_profile({}, {'robot_joint_limit_profile':J16_360_PROFILE})
        check_snapshot_limit_profile({'robot_joint_limit_profile':J16_360_PROFILE},
                                     {'robot_joint_limit_profile':J16_360_PROFILE})
        with self.assertRaisesRegex(ValueError,'不匹配'):
            check_snapshot_limit_profile({'robot_joint_limit_profile':J16_360_PROFILE},{})
        with self.assertRaisesRegex(ValueError,'未知'):
            reference_limits_for_profile(names,old,'all_joints_360')


if __name__=='__main__':unittest.main(verbosity=2)
