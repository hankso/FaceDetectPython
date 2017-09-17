from __future__ import print_function
import keras, os, time, cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.utils import to_categorical



class Detector(object):
    
    def __init__(self):
        self.trained = False
        self.result = 'untrained'
        self._print_summary = True
        
    def _read_data(self):
        #check_data
        if not os.path.exists('./photos'):
            print("\nFolder './photos' not exists.")
            return False
        
        folder_num = os.listdir('./photos').__len__()
        if folder_num > 2:
            
            self.data = []
            self.name_list = []
            
            for i in os.listdir('./photos'):
                path = './photos/' + i
                images = [_ for _ in os.listdir(path) if _[-3:] in ['jpg','png']]
                
                if images.__len__() < 5:
                    print('\nFor higher classification accuracy, model needs at ' + 
                          'least 5 photos for %s stored in %s' % (i, path))
                    print('Skip ' + i)
                    continue
                
                print('Found %d images in %s for \33[1m%s\33[0m' \
                      % (images.__len__(), path, i))
                
                if folder_num < 5:
                    # too many people, waste time
                    print("Reading data of %s ... " % i)
                    self.do_nothing(1)
                
                #read_data
                self.name_list.append(i)
                self.data.append([])
                for image in images:
                    self.data[-1].append(cv2.imread(os.path.join(path, image)))
            
            self.cls_num = self.name_list.__len__()
            return True if self.cls_num else False
        
        else:
            print('\nNo enough images or person for training in folder ./photos')
            return False
        
        
    def _build_model(self, opt, loss, met, cls_num):
        self.model = keras.models.Sequential()
        
        self.model.add(keras.layers.Conv2D(input_shape = self.data.shape[1:],
                                           filters = 64,
                                           kernel_size = 3,
                                           strides = 1,
                                           padding = 'same',
                                           activation = 'relu'))
        self.model.add(keras.layers.Conv2D(64, 3, activation = 'relu'))
        self.model.add(keras.layers.MaxPooling2D(pool_size = 2))
        self.model.add(keras.layers.Dropout(0.25))
        
        self.model.add(keras.layers.Conv2D(32,
                                           3,
                                           padding = 'same',
                                           activation = 'relu'))
        self.model.add(keras.layers.Conv2D(32, 3, activation = 'relu'))
        self.model.add(keras.layers.MaxPooling2D())
        self.model.add(keras.layers.Dropout(0.25))
        
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(256, activation = 'relu'))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(cls_num))
        self.model.add(keras.layers.Activation('softmax'))
        
        self.model.compile(optimizer = opt,
                           loss = loss,
                           metrics = met)
        
        
    def _preprocessing(self, src):
        labels = []
        images = []
        for name_index, imgs in enumerate(src):
            for img_index, img in enumerate(imgs):
                labels.append(name_index)
                
                h, w = img.shape[:2]
                top = bottom = (w-h)/2 if h < w else 0
                left = right = (h-w)/2 if w < h else 0
                img = cv2.copyMakeBorder(img,
                                         top,
                                         bottom,
                                         left,
                                         right,
                                         cv2.BORDER_CONSTANT)
                img = cv2.resize(img, dsize = (128, 128))
                
                images.append(img)
                
                
        return np.array(images, dtype='float32')/255, to_categorical(labels, self.cls_num)
        
        
    def train(self,
              loss = 'categorical_crossentropy',
              optimizer = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
              metrics = ['accuracy'],
              batch_size = 32,
              epochs = 10,
              verbose = 1,
              generator = False,
              use_saved_model = False):
        
        if self.trained:
            if len(os.listdir('./photos')) <= self.cls_num:
                print('\nAlready using the newest model!\nOffer more data to re-train.')
                return
            
        if os.path.exists('./models') and use_saved_model:
            self.model = keras.models.load_model(os.listdir('./models')[-1])
            self.trained = True
            self.result = 'unused'
            return
            
        
        if  self._read_data():
            
            self.data, self.label = self._preprocessing(self.data)
            
            if verbose:
                print('\nSuccessfully load data: ', end = '')
                print(self.data.shape)
                print('\nNow start build model... ')
                self.do_nothing(2)
            
            self._build_model(optimizer, loss, metrics, self.cls_num)
            
            if verbose:
                print('\nSuccessfully built model.')
                print('\nModel summary:')
                print(self.model.summary())
            
            rst = raw_input('Start training? [Y/n] ')
            if rst in ['n','N']:
                print('Abort')
                return
            
            print('\ntraining...')
            self.do_nothing(2)
            if generator:
                data_generator = IDG(featurewise_center=False,
                                     samplewise_center=False,
                                     featurewise_std_normalization=False,
                                     samplewise_std_normalization=False,
                                     zca_whitening=False,
                                     rotation_range=15,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     horizontal_flip=True,
                                     vertical_flip=True)
                
                # Required for featurewise_center,
                #  featurewise_std_normalization
                #   and zca_whitening.
                #data_generator.fit(self.data)
                
                self.model.fit_generator(data_generator.flow(self.data,
                                                             self.label,
                                                             batch_size),
                                         steps_per_epoch = self.data.shape[0] / batch_size,
                                         epochs = epochs)
            else:
                self.model.fit(x = self.data,
                               y = self.label,
                               batch_size = batch_size,
                               epochs = epochs,
                               verbose = verbose)
            
            self.trained = True
            self.result = 'unused'
            self.model.save_weights(os.path.join(os.getcwd(),
                                                 'models',
                                                 time.strftime('%Y%m%d%H%M')+'.h5'))
                
            print('\33[1mSuccessfully trained the model!\33[0m')
            return
        
        print('\nError occurs when trying to train the model. Abort')
        
    def detect(self, img):
        img = [self._preprocessing([[img]])[0]]
        
        self.result = self.model.predict_classes(img)
        
        #TODO confidency
        #rst_label = to_categorical(self.result, self.cls_num)
        #return self.model.evaluate(img, rst_label)[-1], self.name_list[int(self.result)]
        return 1, self.name_list[int(self.result)]
        
        
    def save_photo(self, cam, photo_num = 10, duration = 2):
        name = raw_input("\nPlease input name of this people in the frame or cancel [Name/n]: ")
        if name == 'n':
            print('\33[1mAbort\33[0m')
            return
        name = './photos/' + name
        if not os.path.exists('./photos'):
            os.mkdir('./photos')
        if not os.path.exists(name):
            os.mkdir(name)
        print('\nOK! Now it will start saving a photo every %d seconds, you have 5 seconds to prepare your poses.' % duration)
        self.do_nothing()
        for _ in range(photo_num):
            time.sleep(1 if duration < 1 else duration)
            filename = os.path.join(os.getcwd(),
                                    name,
                                    time.strftime('%Y%m%d%H%M%S') + '.jpg')
            cv2.imwrite(filename, cam.read()[1])
            print('\33[1mSaved:\33[0m ' + filename)
            
    def do_nothing(self, times = 5, mode = 'state_speed'):
        
        times = 5 if not isinstance(times, (int,float)) else int(times)
        
        if mode == 'state_speed':
            rate = 3
            for i in range(rate*times):
                print('\r' + '>'*i + '~'*(rate * times - i), end = '')
                time.sleep(1.0/rate)
            print('\r' + '>'*rate*times)
            time.sleep(1)
            
        elif mode == 'state_length':
            for i in range(2 * times):
                time.sleep(1)
                print('', end = '')
                #TODO s
        else:
            print('\nUnvalid mode! Mode should be one of state_speed or state_length.')
            
if __name__ == '__main__':
    test = Detector()
    test.train()

