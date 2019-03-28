from model import *
import keras.backend as K
from keras.optimizers import Adam
from data_generator import image_generator
from config import *
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

train_step_per_epoch = len(os.listdir(image_source_dir + 'trainA')) / batch_size + 1
test_step_per_epoch = len(os.listdir(image_source_dir + 'testA')) / batch_size + 1
train_image_generator = image_generator(image_source_dir + 'trainA',
                                        image_source_dir + 'trainB', batch_size=batch_size,
                                        shuffle=True)
test_image_generator = image_generator(image_source_dir + 'testA',
                                       image_source_dir + 'testB', batch_size=batch_size,
                                       shuffle=False)


#warning,the original metric(acc) and loss function(mse,mae) is defined with axis=-1,because the output is 2D(batch_size*N),in our cases the output is 4D(batch_size*H*W*C), we should adapt them to our case
def acc1(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=[1,2,3])

#for patch discriminator,output is batch_size*H*W*1
def mse_custom(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=[1,2,3])

#for generator,output is batch_size*H*W*3
def mae_custom(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=[1,2,3])

def mape_custom(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),K.epsilon(),None))
    return 100. * K.mean(diff, axis=[1,2,3])

opt1 = Adam(lr=lr)
opt2 = Adam(lr=lr)
opt3 = Adam(lr=lr)
discriminator_A = get_discriminator(n_layers=int(np.log(downsample)/np.log(2)),instance_norm=use_instance_norm,name='discriminator_a')
discriminator_B = get_discriminator(n_layers=int(np.log(downsample)/np.log(2)),instance_norm=use_instance_norm,name='discriminator_b')
discriminator_A.compile(optimizer=opt1, loss=mse_custom, metrics=[acc1])
discriminator_B.compile(optimizer=opt2, loss=mse_custom, metrics=[acc1])
print(discriminator_B.summary())
generator_A2B = get_generator(name='a2b',instance_norm=use_instance_norm)
generator_B2A = get_generator(name='b2a',instance_norm=use_instance_norm)
# generator_A2B.compile(optimizer='Adam', loss='mae', metrics=['mean_absolute_percentage_error'])
print(generator_A2B.summary())
generator_train = get_generator_training_model(generator_A2B, generator_B2A, discriminator_A, discriminator_B)
print(generator_train.summary())
generator_train.compile(optimizer=opt3, loss=[mae_custom, mae_custom, mae_custom, mae_custom, mse_custom, mse_custom ],
                        metrics=[mape_custom],
                        loss_weights=[5, 5, 10, 10, 1, 1])
#generator_train.compile(optimizer=opt3, loss=[mae_custom, mae_custom, mse_custom, mse_custom, ],
#                        metrics=[mape_custom],
#                        loss_weights=[10, 10, 1, 1])
if os.path.exists(combined_filepath):
    generator_train.load_weights(combined_filepath, by_name=True)
    generator_A2B.load_weights(generator_a2b_filepath, by_name=True)
    generator_B2A.load_weights(generator_b2a_filepath, by_name=True)
    print('weights loaded!')

def imsave(img,filename):
    img = (img + 1) * 127.5
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(filename)


real = np.ones((batch_size, image_size/downsample, image_size/downsample, 1))
fake = np.zeros((batch_size, image_size/downsample, image_size/downsample, 1))
best_loss = 1000

for i in range(epoch):
    train_step = 0
    for imgA, imgB in train_image_generator:
        train_step += 1
        if train_step > train_step_per_epoch:
            test_step = 0
            total_loss = 0
            total_mape = 0
            for imgA, imgB in test_image_generator:
                test_step += 1
                if test_step > test_step_per_epoch:
                    break
                # print generator_train.metrics_names
                #gloss, _, _, _, _, mape1, mape2, mape3, mape4 = generator_train.test_on_batch(
                #    [imgA, imgB], [imgA, imgB, real, real])
                gloss, _, _, _, _, _, _, mape1, mape2, mape3, mape4, mape5, mape6 = generator_train.test_on_batch([imgA, imgB], [imgA, imgB,imgA, imgB,  real, real])
                total_loss += gloss
                #total_mape += sum([mape1, mape2, mape3, mape4])
                total_mape += sum([mape1, mape2, mape3, mape4, mape5, mape6])
            print('epoch:{} test loss g:{} \n   test mape:{}'.format(i + 1, total_loss / (test_step - 1),
                                                                     total_mape / (test_step - 1)))
            if total_loss / (test_step - 1) < best_loss:
                print('test loss improved from {} to {}'.format(best_loss, total_loss / (test_step - 1)))
                best_loss = total_loss / (test_step - 1)
            generator_train.save_weights(combined_filepath, overwrite=True)
            generator_A2B.save_weights(generator_a2b_filepath, overwrite=True)
            generator_B2A.save_weights(generator_b2a_filepath, overwrite=True)
            break
        #print(discriminator_A.summary())
        #print(discriminator_B.summary())
        fakeB = generator_A2B.predict(imgA)
        fakeA = generator_B2A.predict(imgB)
        if debug:
            fn=output_dir + str(i + 1) + '_' + str(train_step) + '_fake_a2b.png'
            imsave(fakeB[0],fn)
            #print("{} saved".format(fn))
            fn=output_dir + str(i + 1) + '_' + str(train_step) + '_real_a.png'
            imsave(imgA[0],fn)
            #print("{} saved".format(fn))
            fn=output_dir + str(i + 1) + '_' + str(train_step) + '_fake_b2a.png'
            imsave(fakeA[0],fn)
            #print("{} saved".format(fn))
            fn=output_dir + str(i + 1) + '_' + str(train_step) + '_real_b.png'
            imsave(imgB[0],fn)
            #print("{} saved".format(fn))
            # print('realB:', imgB[0], imgB.shape)
            # print descriminator.trainable
            #d_fakeA = discriminator_A.predict(fakeA)
            #d_realA = discriminator_A.predict(imgA)
            #d_fakeB = discriminator_B.predict(fakeB)
            #d_realB = discriminator_B.predict(imgB)
            #print('d_real_A:', np.squeeze(d_realA[0]), d_realA.shape)
            #print('d_fake_A:', np.squeeze(d_fakeA[0]))
            #print('d_real_B:', np.squeeze(d_realB[0]))
            #print('d_fake_B:', np.squeeze(d_fakeB[0]))
        discriminator_A.trainable = True
        discriminator_B.trainable = False
        generator_A2B.trainable = False
        generator_B2A.trainable = False
        loss_fakeA, fake_accA = discriminator_A.train_on_batch(fakeA, fake)
        loss_realA, real_accA = discriminator_A.train_on_batch(imgA, real)
        discriminator_A.trainable = False
        discriminator_B.trainable = True
        loss_fakeB, fake_accB = discriminator_B.train_on_batch(fakeB, fake)
        loss_realB, real_accB = discriminator_B.train_on_batch(imgB, real)
        print('epoch:{} train step:{}, loss d_fake:{:.2}, loss d_real:{:.2}, fake_acc:{:.2}, real_acc:{:.2}'.format(i + 1, train_step,loss_fakeA + loss_fakeB,loss_realA + loss_realB,(fake_accA + fake_accB) * 0.5,(real_accA + real_accB) * 0.5))
        # print descriminator.metrics_names
        if i+1<pretrain_epoch or (train_step>pretrain_step_start and train_step<pretrain_step_end):
        #if (i+1)%2==1 or i<pretrain_epoch:# or (train_step>pretrain_step_start and train_step<pretrain_step_end):
            continue
        discriminator_A.trainable = False
        discriminator_B.trainable = False
        generator_A2B.trainable = True
        generator_B2A.trainable = True
        #print(generator_train.summary())
        # print generator_train.metrics_names
        #loss = generator_train.train_on_batch([imgA, imgB], [imgA, imgB, imgA, imgB, real, real])
        #loss = generator_train.train_on_batch([imgA, imgB], [imgA, imgB, real, real])
        loss = generator_train.train_on_batch([imgA, imgB], [imgA, imgB, imgA, imgB, real, real])
        # print descriminator.trainable
        # print('epoch:{} train step:{} loss cycle:{:.2} loss fool:{:.2}'.format(i + 1, train_step,loss[1] + loss[2],loss[3] + loss[4]))
        print('epoch:{} train step:{} loss identity:{:.2} loss cycle:{:.2} loss gan:{:.2}'.format(i + 1, train_step,loss[1] + loss[2],loss[3] + loss[4],loss[5] + loss[6]))
