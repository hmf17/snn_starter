from PWCSNet import *
from dataloader import *
import imageio

# To run the test file, you should set the batch_size to 1.

checkpoint_path = 'checkpoint'
dataset_path = 'dataset/test'

# load model
def model_load():
    model = PWCSNet()
    checkpoint_root = checkpoint_path+'/model_DVS_Optical_Flow-2020-05-30-18-14-49.t7'
    """
    You probably saved the model using nn.DataParallel, 
    which stores the model in module, and now you are trying to load it without . 
    You can either add a nn.DataParallel temporarily in your network for loading purposes, 
    or you can load the weights file, create a new ordered dict without the module prefix, and load it back.
    """
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(checkpoint_root).items()})
    return model

dvs_1_path = 'dataset\\training\\final\\cave_4\\dvs_0001'
dvs_2_path = 'dataset\\training\\final\\cave_4\\dvs_0002'
flow_path = 'dataset\\training\\flow\\cave_4\\frame_0001.flo'

if __name__=='__main__':

    model = model_load()
    model = model.to(device)

    dvs_1 = read_dvs(dvs_1_path)
    dvs_2 = read_dvs(dvs_2_path)
    dvs = torch.stack([dvs_1, dvs_2])
    dvs = torch.unsqueeze(dvs, 0).to(device).float()
    flow_gt = read_flow(flow_path)

    # image = torch.load(os.path.join(dataset_path, "dvs.data"))
    # flow_gt = torch.load(os.path.join(dataset_path, "flow.data"))

    flow_pred_list = model(dvs)

    with torch.no_grad():
        flow_gt = flow_gt.permute([1,0,2])
        img = vis_flow(np.array(flow_gt.cpu()))
        imageio.imsave('predict_result/flow_gt.png', img)
        for i, flow_pred in enumerate(flow_pred_list):
            flow_pred = flow_pred.squeeze().permute([2, 1, 0])
            img = vis_flow(np.array(flow_pred.cpu()))
            imageio.imsave('predict_result/flow_pred_'+ str(i) +'.png', img)


        flow_pred = flow_pred_list[0].cpu().squeeze().permute([2,1,0])
        flow_pred = bilinear_interpolation(np.array(flow_pred), (np.array(flow_gt).shape)[:-1], align_corners=False)


        img = vis_flow(flow_pred)
        imageio.imsave('predict_result/flow_pred.png', img)