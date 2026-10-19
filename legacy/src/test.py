# Example file showing a circle moving on screen
import datetime
import math
import random
import time
from typing import List
import pygame

from beat import get_beat_list


# pygame setup
pygame.init()

audio_file = '../Beat Thee.mp3'
beat_list = get_beat_list(audio_file)
pygame.mixer.music.load(audio_file)
pygame.mixer.music.play(-1)

screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0
current_index = 0

class BaseObject(object):
    def __init__(self,screen,color,pos:pygame.Vector2,speed:list):
        self.screen = screen
        self.color = color
        self.pos = pos
        self.speed = speed
        self.is_stop = False
    
    def draw(self):
        pygame.draw.rect(self.screen, self.color, self.get_rect())

    def get_rect(self)->pygame.Rect:
        raise NotImplementedError("Subclasses must implement this method.")

    def set_speed(self,speed:list):
        self.speed = speed

    def rever_speed(self,x=True,y=True):
        if x:
            self.speed[0] = -self.speed[0]
        if y:
            self.speed[1] = -self.speed[1]

    def move(self):
        if self.is_stop:
            return
        self.pos.x += self.speed[0] * dt
        self.pos.y += self.speed[1] * dt

    def run(self):
        self.is_stop = False
    
    def stop (self):
        self.is_stop = True
    
    def get_speed(self):
        return self.speed.copy()
    
    def check_is_collision(self,obj)->int:
        # 未撞击返回0 撞击返回方向 1,2,3,4 左右上下
        self_rect = self.get_rect()
        other_rect = obj.get_rect()
        if self_rect.colliderect(other_rect):
            intersection = self_rect.clip(other_rect)
            if intersection.width > intersection.height:
                if self_rect.left < other_rect.left:
                    return 4
                else:
                    return 3
            else:
                if self_rect.top < other_rect.top:
                    return 2
                else:
                    return 1
        return 0


class Ball(BaseObject):
    # def __init__(self,screen,color,pos:pygame.Vector2,speed:list,radius):
    #     super().__init__(screen,color,pos,speed)
    #     self.radius = radius

    # def draw(self):
    #     pygame.draw.circle(self.screen, self.color, self.pos, self.radius)
    def __init__(self,screen,color,pos:pygame.Vector2,speed:list,width,height):
        super().__init__(screen,color,pos,speed)
        self.width = width
        self.height = height

    def draw(self):
        pygame.draw.rect(self.screen, self.color, self.get_rect())

    def get_rect(self):
        return pygame.Rect(self.pos.x, self.pos.y, self.width, self.height)
    
    def check_is_out_edge(self)->list:
        # if self.pos.x - self.radius < screen.get_width()/4 or self.pos.x + self.radius > self.screen.get_width()/4*3:
        #     return True
        # if self.pos.y - self.radius < screen.get_height()/4 or self.pos.y + self.radius > self.screen.get_height()/4*3:
        #     return True
        # 1 2 3 4 左右上下
        out_place = []
        if self.pos.x  < screen.get_width()/4:
            out_place.append(1)
        if self.pos.x > self.screen.get_width()/4*3:
            out_place.append(2)
        if self.pos.y  < screen.get_height()/4:
            out_place.append(3)
        if self.pos.y > self.screen.get_height()/4*3:
            out_place.append(4)
        return out_place
        
        
    def check_is_to_center(self)->bool:
        center_point = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
        if (abs(self.pos.x-center_point.x) >= abs(self.pos.x+self.speed[0]-center_point.x) and abs(self.pos.y-center_point.y) >= abs(self.pos.y+self.speed[1]-center_point.y)):
            return True
        # if (((self.pos.x-center_point.x) >= 0) and self.speed[0] <= 0) or (((self.pos.y+center_point.y) >= 0) and self.speed[1] <= 0):
        #     return True
        return False
    

    
class Wall(BaseObject):
    def __init__(self,screen,color,pos:pygame.Vector2,speed:list,width,height):
        super().__init__(screen,color,pos,speed)
        self.width = width
        self.height = height

    def draw(self):
        pygame.draw.rect(self.screen, self.color, self.get_rect())

    def move(self,speed:list):
        if self.is_stop:
            return
        self.pos.x += speed[0] * dt
        self.pos.y += speed[1] * dt

    def get_rect(self):
        return pygame.Rect(self.pos.x, self.pos.y, self.width, self.height)

class WallList(list):
    def __init__(self,element_type):
        self.element_type = element_type
        self.is_stop = True
        self.speed = [0,0]
    
    def append(self, item):
        if not isinstance(item, Wall):
            raise TypeError("Invalid element type. Expected Wall, got {type(item)}")
        super().append(item)

    def draw(self):
        for wall in self:
            wall.draw()

    def move(self):
        if self.is_stop:
            return
        for wall in self:
            wall.move(self.speed)

    def get_point_part_num(self,pos:pygame.Vector2)->int:
        #获取所有wall相对与点的位置数量
        nums = [0,0,0,0] #左上右上左下右下↖️↗️↙️↘️
        for wall in self:
            if wall.pos.x>pos.x and wall.pos.y>pos.y:
                nums[3] += 1
            elif wall.pos.x<pos.x and wall.pos.y>pos.y:
                nums[2] += 1
            elif wall.pos.x>pos.x and wall.pos.y<pos.y:
                nums[1] += 1
            else:
                nums[0]+=1
        return nums
    
    def stop(self):
        self.is_stop = True
    def run(self):
        self.is_stop = False
    
    def set_speed(self,speed:list):
        self.speed = speed

    def rever_speed(self,x=True,y=True):
        if x:
            self.speed[0] = -self.speed[0]
        if y:
            self.speed[1] = -self.speed[1]
    

wall = Wall(screen, (255, 0, 255), pygame.Vector2(800, 500), [0,0], 20, 40)
ball = Ball(screen, (255, 0, 255), pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2), [400,400], 50,50)

wall_list = WallList(Wall)
wall_list.append(wall)

frame_count = 0

def get_sorted_indexes(nums):
    index_dict = {}
    for i, num in enumerate(nums):
        if num in index_dict:
            index_dict[num].append(i)
        else:
            index_dict[num] = [i]
    
    sorted_indexes = []
    for num in sorted(index_dict.keys()):
        sorted_indexes.extend(index_dict[num])
    
    return sorted_indexes



def where_generage_wall(ball:Ball,wall_list:WallList):
    #判断生成wall的位置
    #选择最少的两个方向
    
    #获取小球朝向
    speed = ball.get_speed()
    forward = 0  #0123↖️↗️↙️↘️
    now_user = [] #左右上下0123
    if speed[0] > 0:
        if speed[1] > 0:
            forward = 3
            not_user = [0,1]
        else:
            forward = 1
            not_user = [0,3]
    else:
        if speed[1] > 0:
            forward = 2
            not_user = [1,2]
        else:
            forward = 0
            not_user = [1,3]
    print(not_user)
    part_num_list = wall_list.get_point_part_num(ball.pos)
    print(part_num_list)
    rank_arr = get_sorted_indexes(part_num_list)
    print(rank_arr)
    if random.random() < 0.5:
        rank_arr.reverse()
    for check_index in rank_arr:
        if check_index == 0:
        #左上
            if random.random() < 0.5:
                first = 1
                second = 3
            else:
                first = 3
                second = 1
            if first in not_user:
                if second in not_user:
                    continue
                return generagte_wall_pos(ball,second)
            return generagte_wall_pos(ball,first)

        elif check_index == 1:
            #右上
            if random.random() < 0.5:
                first = 0
                second = 3
            else:
                first = 3
                second = 1
            if first in not_user:
                if second in not_user:
                    continue
                return generagte_wall_pos(ball,second)
            return generagte_wall_pos(ball,first)
        elif check_index == 2:
            #左下
            if random.random() < 0.5:
                first = 1
                second = 2
            else:
                first = 2
                second = 1
            if first in not_user:
                if second in not_user:
                    continue
                return generagte_wall_pos(ball,second)
            return generagte_wall_pos(ball,first)
        else:
            #右下
            if random.random() < 0.5:
                first = 0
                second = 2
            else:
                first = 2
                second = 0
            if first in not_user:
                if second in not_user:
                    continue
                return generagte_wall_pos(ball,second)
            return generagte_wall_pos(ball,first)
        


def generagte_wall_pos(ball:Ball,forward=int):#0123左右上下
    print(forward)
    if forward==0:
        wall_pos = pygame.Vector2(ball.pos.x-10,ball.pos.y)
    elif forward==1:
        wall_pos = wall_pos = pygame.Vector2(ball.pos.x+ball.width,ball.pos.y)
    elif forward==2:
        wall_pos = pygame.Vector2(ball.pos.x,ball.pos.y-10)
    else:
        wall_pos = pygame.Vector2(ball.pos.x,ball.pos.y+ball.height)
    return wall_pos

    
start_time = time.time()

while running:
    # frame_count += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # fill the screen with a color to wipe away anything from last frame
    out_edge_place = ball.check_is_out_edge()
    screen.fill("purple")
    
    minutes,current_time = divmod(time.time()-start_time, 60)
    # print(current_time)
    if current_time >= beat_list[current_index]:
        print(f'check:{current_time},time:{beat_list[current_index]}')
        current_index+=1

        pos = where_generage_wall(ball,wall_list)
        random_color = (random.randint(0,255), random.randint(0,255),random.randint(0,255))
        new_wall = Wall(screen,random_color,pos,[0,0],10,10)

        # speed = ball.get_speed()
        # # random_color = (random.randint(0,255), random.randint(0,255),random.randint(0,255))
        # # wall_pos = pygame.Vector2(ball.pos.x + 10,ball.pos.y + 10)
        # # new_wall = Wall(screen,random_color,wall_pos,[0,0],10,10)
        # # 判断是否越界 如果在界外则需要反向生成
        # if out_edge_place:
        #     print('越界了 纠正中')
        #     print(out_edge_place)
        #     random_forward = out_edge_place[random.randint(0,len(out_edge_place)-1)]
        #     if 1 == random_forward:
        #         print('1')
        #         wall_pos=generagte_wall_pos(ball,left=True)
        #     elif 2 == random_forward:
        #         print(2)
        #         wall_pos=generagte_wall_pos(ball,right=True)
        #         new_wall = Wall(screen,random_color,wall_pos,pygame.Vector2(0,0),10,10)
        #     elif 3 == random_forward:
        #         print(3)
        #         wall_pos=generagte_wall_pos(ball,up=True)
        #     else:
        #         print(4)
        #         wall_pos=generagte_wall_pos(ball,down=True)
        #     new_wall = Wall(screen,random_color,wall_pos,[0,0],10,10)
        # else:
        #     # 在ball的方向随机生成一个新的wall
        #     random_color = (random.randint(0,255), random.randint(0,255),random.randint(0,255))
        #     if random.random() > 0.2:
        #         print(speed)
        #         #x方向
        #         if speed[0] < 0:
        #             print('生成left')
        #             wall_pos=generagte_wall_pos(ball,left=True)
        #         else:
        #             wall_pos=generagte_wall_pos(ball,right=True)
        #         new_wall = Wall(screen,random_color,wall_pos,pygame.Vector2(0,0),10,10)
        #     else:
        #         if speed[1] < 0:
        #             wall_pos=generagte_wall_pos(ball,up=True)
        #         else:
        #             wall_pos=generagte_wall_pos(ball,down=True)
        #         new_wall = Wall(screen,random_color,wall_pos,[0,0],10,10)
        wall_list.append(new_wall)
    for wall in wall_list:
        is_collision = ball.check_is_collision(wall) 
        if is_collision:
            if is_collision in [1,2]:
                ball.rever_speed(True,False)
            else:
                ball.rever_speed(False,True)
            break
    
    if ball.check_is_to_center():
        print('go_center')
        wall_list.stop()
        ball.run()
    else:
        if out_edge_place==[]:
            print('out')
            #如果小球出界，则小球stop障碍反向运动
            wall_list.set_speed(ball.get_speed())
            wall_list.rever_speed()
            wall_list.run()
            ball.stop()
        else:
            print('not_out')
            ball.run()
            wall_list.stop()
    
    ball.move()
    wall_list.move()
    ball.draw()
    wall_list.draw()

    font = pygame.font.Font(None, 36)
    text = font.render("Frame: {}".format(frame_count), True, (255, 255, 255))
    screen.blit(text, (50, 50))

    pygame.display.flip()


    dt = clock.tick(120) / 1000
    frame_count += 1


pygame.quit()