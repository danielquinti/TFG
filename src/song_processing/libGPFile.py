#!/usr/bin/python3
import itertools
FILENAME="ACDC - Can_'t stand still.gp4"
FILENAME="Harper, Ben - Alone (Live).gp4"
#FILENAME="Beatles (The) - All Together Now.gp3"
#FILENAME="Beatles (The) - All Together Now-tux.gp4"

class Reader:
        def __init__(self, filename):
                self.file=open(filename, "rb")
                self.CHUNKSIZE=4
                self.chunk=[]
                self.pos=self.CHUNKSIZE+1000
                self.globalPos=0

        def close(self):
                self.file.close()

        def _readChunk(self):
                self.chunk=self.file.read(self.CHUNKSIZE)
                if not self.chunk:
                        raise EOFError()
                self.pos=0
        def readBytes(self, nBytes):
                total=b''
                while nBytes>0:
                        if self.pos>=len(self.chunk):
                                self._readChunk()
                        nRead=nBytes
                        if (self.pos+nRead>len(self.chunk)):
                                nRead=len(self.chunk)-self.pos
                        total+=self.chunk[self.pos:self.pos+nRead]
                        self.pos+=nRead
                        self.globalPos+=nRead
                        nBytes-=nRead
                return total
        def readString(self):
                length=int.from_bytes(self.readBytes(1), byteorder="big")
                s=self.readBytes(length)
                return s.decode("ISO-8859-1")

        def readInt(self,signed=False):
                return int.from_bytes(self.readBytes(4), byteorder="little", signed=signed)
        def readByte(self):
                return self.readBytes(1)
        def skipBytes(self, n):
                self.readBytes(n)
        def printPos(self):
                print("Position -> %s"%hex(self.globalPos))

class Writer:
        def __init__(self, filename):
                self.file=open(filename, "wb")

        def writeBytes(self,b):
                self.file.write(b)

        def skipBytes(self,n):
                self.file.write(bytes([0])*n)

        def writeString(self,s):
                self.writeBytes(bytes([len(s)]))
                self.writeBytes(s.encode("utf8"))

        def writeInt(self, v, signed=False):
                self.writeBytes(v.to_bytes(4, byteorder="little", signed=signed))

        def writeMask(self, mask):
                factor=1
                result=0
                for m in mask:
                        if m:
                                result+=factor
                        factor*=2
                self.writeBytes(bytes([result]))

        def close(self):
                self.file.close()


class GPFile:
        class GPHeader:
                class GPMidiChannel:
                        def __init__(self,instrument,volume,balance,chorus,reverb,phaser,tremolo,blank1,blank2):
                                self.instrument=instrument
                                self.volume=volume
                                self.balance=balance
                                self.chorus=chorus
                                self.reverb=reverb
                                self.phaser=phaser
                                self.tremolo=tremolo
                                self.blank1=blank1
                                self.blank2=blank2

                        def print(self, prefix):
                                print(prefix+"Instrument: %d"%self.instrument)
                                print(prefix+"Volume: %s"%self.volume)
                                print(prefix+"Balance: %s"%self.balance)
                                print(prefix+"Chorus: %s"%self.chorus)
                                print(prefix+"Reverb: %s"%self.reverb)
                                print(prefix+"Phaser: %s"%self.phaser)
                                print(prefix+"Tremolo: %s"%self.tremolo)
                                print(prefix+"Blank1: %s"%self.blank1)
                                print(prefix+"Blank2: %s"%self.blank2)

                        @staticmethod
                        def read(r):
                                instrument=r.readInt()
                                volume=r.readByte()
                                balance=r.readByte()
                                chorus=r.readByte()
                                reverb=r.readByte()
                                phaser=r.readByte()
                                tremolo=r.readByte()
                                blank1=r.readByte()
                                blank2=r.readByte()
                                return GPFile.GPHeader.GPMidiChannel(instrument,volume,balance,chorus,reverb,phaser,tremolo,blank1,blank2)

                        def write(self,w):
                                w.writeInt(self.instrument)
                                w.writeBytes(self.volume)
                                w.writeBytes(self.balance)
                                w.writeBytes(self.chorus)
                                w.writeBytes(self.reverb)
                                w.writeBytes(self.phaser)
                                w.writeBytes(self.tremolo)
                                w.writeBytes(self.blank1)
                                w.writeBytes(self.blank2)

                def __init__(self, version, title, subtitle, interpreter, album, copyright, author, tabauthor, instructional, note, tripletFeel, tempo, key, octave, midiChannels, numMeasures, numTracks, lyrics, lyricsTrack):
                        self.version=version
                        self.title=title
                        self.subtitle=subtitle
                        self.interpreter=interpreter
                        self.album=album
                        self.copyright=copyright
                        self.author=author
                        self.tabauthor=tabauthor
                        self.instructional=instructional
                        self.note=note
                        self.tripletFeel=tripletFeel
                        self.tempo=tempo
                        self.key=key
                        self.octave=octave
                        self.midiChannels=midiChannels
                        self.numMeasures=numMeasures
                        self.numTracks=numTracks
                        self.lyrics=lyrics
                        self.lyricsTrack=lyricsTrack

                def print(self, prefix='\t', full=False):
                        print(prefix+"Version: %s"%self.version)
                        print(prefix+"Title: %s"%self.title)
                        print(prefix+"Subtitle: %s"%self.subtitle)
                        print(prefix+"Interpreter: %s"%self.interpreter)
                        print(prefix+"Album: %s"%self.album)
                        print(prefix+"Copyright: %s"%self.copyright)
                        print(prefix+"Author: %s"%self.author)
                        print(prefix+"Tab author: %s"%self.tabauthor)
                        print(prefix+"Instructional: %s"%self.instructional)
                        print(prefix+"Note:")
                        for l in self.note:
                                print(prefix+"\t%s"%l)
                        print(prefix+"Triplet feel: %d"%self.tripletFeel)
                        print(prefix+"Tempo: %d"%self.tempo)
                        print(prefix+"Key: %d"%self.key)
                        print(prefix+"Octave: %d"%self.octave)
                        print(prefix+"Measures: %d"%self.numMeasures)
                        print(prefix+"Tracks: %d"%self.numTracks)
                        if self.lyricsTrack:
                                print(prefix+"Lyrics track: %d"%self.lyricsTrack)
                        if self.lyrics:
                                print(prefix+"Lyrics:")
                                for l in self.lyrics:
                                        print(prefix+"\t%s"%l)
                        if not full:
                                print(prefix+"(Ommiting midi channels)")
                        else:
                                for p in range(len(self.midiChannels)):
                                        port=self.midiChannels[p]
                                        for c in range(len(port)):
                                                print(prefix+"Port %d / Channel %d:"%(p,c))
                                                port[c].print(prefix+"\t")

                @staticmethod
                def read(r):
                        #Version
                        version=r.readString()
                        r.skipBytes(30-len(version))

                        #Information about the piece
                        int_size=r.readInt()
                        title=r.readString()

                        int_size=r.readInt()
                        subtitle=r.readString()

                        int_size=r.readInt()
                        interpreter=r.readString()

                        int_size=r.readInt()
                        album=r.readString()

                        int_size=r.readInt()
                        copyright=r.readString()

                        int_size=r.readInt()
                        author=r.readString()

                        int_size=r.readInt()
                        tabauthor=r.readString()

                        int_size=r.readInt()
                        instructional=r.readString()
                        
                        note=[]
                        note_lines=r.readInt()
                        for i in range(note_lines):
                                int_size=r.readInt()
                                line=r.readString()
                                note.append(line)

                        tripletFeel=int.from_bytes(r.readByte(), byteorder="big")!=0

                        #Lyrics
                        lyrics=[]
                        lyricsTrack=None
                        if "3.0" not in version:
                                lyricsTrack=r.readInt()
                                for i in range(10):
                                        chars=r.readInt()
                                        #l=r.readBytes(chars).decode("utf8")
                                        #if l.strip():
                                        #        lyrics.append(l)

                        #Other info
                        tempo=int.from_bytes(r.readByte(), byteorder="big")

                        key=int.from_bytes(r.readByte(), byteorder="big")

                        octave=0
                        if "3.0" not in version:
                                octave=int.from_bytes(r.readByte(), byteorder="big")

                        midiChannels=[]
                        #r.printPos()
                        for i in range(4):
                                port=[]
                                for j in range(16):
                                        port.append(GPFile.GPHeader.GPMidiChannel.read(r))
                                midiChannels.append(port)
                        #TODO??
                        r.skipBytes(6)
                        numMeasures=r.readInt()
                        numTracks=r.readInt()
                        return GPFile.GPHeader(version, title, subtitle, interpreter, album, copyright, author, tabauthor, instructional, note, tripletFeel, tempo, key, octave, midiChannels, numMeasures, numTracks, lyrics, lyricsTrack)
                def write(self,w):
                        VERSION="FICHIER GUITAR PRO v4.00"
                        w.writeString(VERSION)

                        w.skipBytes(30-len(VERSION))

                        w.writeInt(len(self.title)+1)
                        w.writeString(self.title)

                        w.writeInt(len(self.subtitle)+1)
                        w.writeString(self.subtitle)

                        w.writeInt(len(self.interpreter)+1)
                        w.writeString(self.interpreter)

                        w.writeInt(len(self.album)+1)
                        w.writeString(self.album)

                        w.writeInt(len(self.copyright)+1)
                        w.writeString(self.copyright)

                        w.writeInt(len(self.author)+1)
                        w.writeString(self.author)

                        w.writeInt(len(self.tabauthor)+1)
                        w.writeString(self.tabauthor)

                        w.writeInt(len(self.instructional)+1)
                        w.writeString(self.instructional)

                        w.writeInt(len(self.note))

                        for n in self.note:
                                w.writeInt(len(n)+1)
                                w.writeString(n)

                        w.writeBytes(bytes([self.tripletFeel]))


                        if self.lyricsTrack is not None:
                                w.writeInt(self.lyricsTrack)
                        else:
                                w.writeInt(0)
                        for i in range(10):
                                        w.writeInt(0)

                        w.writeBytes(bytes([self.tempo]))
                        w.writeBytes(bytes([self.key]))
                        w.writeBytes(bytes([self.octave]))

                        for i in range(4):
                                for j in range(16):
                                        self.midiChannels[i][j].write(w)

                        w.skipBytes(6)
                        w.writeInt(self.numMeasures)
                        w.writeInt(self.numTracks)

        #END GPHeader

        class GPMeasure:
                def __init__(self, isDoubleBar=False, tonality=None, marker=None, numberOfAlternate=None, numRepeats=None, isBeginRepeat=False, denominator=None, numerator=None):
                        self.isDoubleBar=isDoubleBar
                        self.tonality=tonality
                        self.marker=marker
                        self.numberOfAlternate=numberOfAlternate
                        self.numRepeats=numRepeats
                        self.isBeginRepeat=isBeginRepeat
                        self.denominator=denominator
                        self.numerator=numerator

                def print(self,prefix):
                        s=""
                        if self.isDoubleBar:
                                s+="DOUBLE "
                        if self.isDoubleBar:
                                s+="BEGIN "
                        if self.tonality:
                                s+="TONALITY:%d "%self.tonality
                        if self.numberOfAlternate:
                                s+="ALT#:%d "%self.numberOfAlternate
                        if self.numRepeats:
                                s+="R#:%d "%self.numRepeats
                        if self.denominator:
                                s+="%d"%self.denominator
                        if self.numerator:
                                s+="/%d"%self.numerator
                        if self.marker:
                                s+=" -- MARKER [rgb(%d,%d,%d)]:\t%s"%self.marker
                        print(prefix+s)

                @staticmethod
                def read(r):
                        header=r.readByte()[0]
                        isDoubleBar=(header & 0b10000000) > 0
                        hasTonality=(header & 0b1000000) > 0
                        hasMarker=(header & 0b100000) > 0
                        numberOfAlternate=(header & 0b10000) > 0
                        endRepeat=(header & 0b1000) > 0
                        beginRepeat=(header & 0b100) > 0
                        hasDenominator=(header & 0b10) > 0
                        hasNumerator=(header & 0b1) > 0

                        if header==0x80: #Skipping TODO - Verify this works as intended
                                header=r.readByte()[0]
                                isDoubleBar=(header & 0b10000000) > 0
                                hasTonality=(header & 0b1000000) > 0
                                hasMarker=(header & 0b100000) > 0
                                numberOfAlternate=(header & 0b10000) > 0
                                endRepeat=(header & 0b1000) > 0
                                beginRepeat=(header & 0b100) > 0
                                hasDenominator=(header & 0b10) > 0
                                hasNumerator=(header & 0b1) > 0

                        data={}

                        if header!=0:
                                if isDoubleBar:
                                        data["isDoubleBar"]=True
                                if beginRepeat:
                                        data["isBeginRepeat"]=True
                                if hasNumerator:
                                        data["numerator"]=int.from_bytes(r.readByte(), byteorder="big")
                                if hasDenominator:
                                        data["denominator"]=int.from_bytes(r.readByte(), byteorder="big")
                                if endRepeat:
                                        data["numRepeats"]=int.from_bytes(r.readByte(), byteorder="big")
                                if numberOfAlternate:
                                        data["numberOfAlternate"]=int.from_bytes(r.readByte(), byteorder="big")
                                if hasMarker:
                                        int_size=r.readInt()
                                        #print("\tINT:\t%d"%int_size)
                                        marker=r.readString()
                                        red=r.readByte()[0]
                                        green=r.readByte()[0]
                                        blue=r.readByte()[0]
                                        white=r.readByte()[0]
                                        data["marker"]=(red,green,blue,marker)
                                if hasTonality:
                                        data["tonality"]=int.from_bytes(r.readByte(), byteorder="big")
                        return GPFile.GPMeasure(**data)

                def write(self,w):
                        w.writeMask([self.numerator is not None, \
                                        self.denominator is not None, \
                                        self.isBeginRepeat, \
                                        self.numRepeats is not None, \
                                        self.numberOfAlternate is not None, \
                                        self.marker is not None, \
                                        self.tonality is not None, \
                                        self.isDoubleBar])

                        if self.numerator is not None:
                                w.writeBytes(bytes([self.numerator]))
                        if self.denominator is not None:
                                w.writeBytes(bytes([self.denominator]))
                        if self.numRepeats is not None:
                                w.writeBytes(bytes([self.numRepeats]))
                        if self.numberOfAlternate is not None:
                                w.writeBytes(bytes([self.numberOfAlternate]))
                        if self.marker is not None:
                                red,green,blue,text=self.marker
                                w.writeInt(len(text)+1)
                                w.writeString(text)
                                w.writeBytes(bytes([red]))
                                w.writeBytes(bytes([green]))
                                w.writeBytes(bytes([blue]))
                                w.skipBytes(1)#White
                        if self.tonality is not None:
                                w.writeBytes(bytes([self.tonality]))

        #END GPMeasure

        class GPTrack:
                def __init__(self, name, numStrings, tuning, port, channel, channelE, numFrets, capo, color, isBanjo=False, is12String=False, isDrums=False):
                        self.name=name
                        self.numStrings=numStrings
                        self.tuning=tuning
                        self.port=port
                        self.channel=channel
                        self.channelE=channelE
                        self.numFrets=numFrets
                        self.capo=capo
                        self.color=color
                        self.isBanjo=isBanjo
                        self.is12String=is12String
                        self.isDrums=isDrums

                def print(self, prefix):
                        s=""
                        if self.isBanjo:
                                s+="BANJO "
                        if self.is12String:
                                s+="12String "
                        if self.isDrums:
                                s+="DRUMS "
                        s+=self.name
                        print(prefix+s)
                @staticmethod
                def read(r):
                        tHeader=r.readByte()[0]
                        isBanjo=(tHeader & 0b100) > 0
                        is12String=(tHeader & 0b10) > 0
                        isDrums=(tHeader & 0b1) > 0

                        data={"isBanjo":isBanjo, "is12String":is12String, "isDrums":isDrums, "name":r.readString()}
                        r.skipBytes(40-len(data["name"]))

                        data["numStrings"]=r.readInt()

                        tuning=[]
                        for j in range(7):
                                tuning.append(r.readInt())
                        data["tuning"]=tuning

                        for n in ["port","channel","channelE","numFrets","capo"]:
                                data[n]=r.readInt()

                        red=r.readByte()[0]
                        green=r.readByte()[0]
                        blue=r.readByte()[0]
                        white=r.readByte()[0]

                        data["color"]=(red,green,blue)
                        return GPFile.GPTrack(**data)

                def write(self,w):
                        w.writeMask([self.isDrums, \
                                    self.is12String, \
                                    self.isBanjo, \
                                    False,
                                    False,
                                    False,
                                    False,
                                    False])

                        w.writeString(self.name)
                        w.skipBytes(40-len(self.name))
                        w.writeInt(self.numStrings)
                        for t in self.tuning:
                                w.writeInt(t)
                        w.writeInt(self.port)
                        w.writeInt(self.channel)
                        w.writeInt(self.channelE)
                        w.writeInt(self.numFrets)
                        w.writeInt(self.capo)

                        (red,green,blue)=self.color
                        w.writeBytes(bytes([red]))
                        w.writeBytes(bytes([green]))
                        w.writeBytes(bytes([blue]))
                        w.skipBytes(1)#White

        #END GPTrack

        class GPBeat:

                class GPChordDiagram:
                        def __init__(self, header, name, baseFret, frets, fifth=None, ninth=None, eleventh=None, numBarres=None, barres=None, ommissions=None, fingering=None, showFingering=None, sharp=None, root=None, cType=None, extra=None, bass=None, dimaug=None, add=None):
                                self.header=header
                                self.sharp=sharp
                                self.root=root
                                self.cType=cType
                                self.extra=extra
                                self.bass=bass
                                self.dimaug=dimaug
                                self.add=add
                                self.name=name
                                self.fifth=fifth
                                self.ninth=ninth
                                self.eleventh=eleventh
                                self.baseFret=baseFret
                                self.frets=frets
                                self.numBarres=numBarres
                                self.barres=barres
                                self.ommissions=ommissions
                                self.fingering=fingering
                                self.showFingering=showFingering


                        @staticmethod
                        def read(r,version):
                                data={}
                                data["header"]=r.readByte()[0]

                                if "3.0" in version:
                                        r.skipBytes(25)
                                else:
                                        data["sharp"]=r.readByte()[0]
                                        r.skipBytes(3)
                                        data["root"]=r.readByte()[0]
                                        data["cType"]=r.readByte()[0]
                                        data["extra"]=r.readByte()[0]
                                        data["bass"]=r.readInt()
                                        data["dimaug"]=r.readInt()
                                        data["add"]=r.readByte()[0]
                                data["name"]=r.readString()
                                r.skipBytes(20-len(data["name"]))
                                if "3.0" in version:
                                        r.skipBytes(14)
                                else:
                                        r.skipBytes(2)
                                        data["fifth"]=r.readByte()[0]
                                        data["ninth"]=r.readByte()[0]
                                        data["eleventh"]=r.readByte()[0]
                                data["baseFret"]=r.readInt()
                                frets=[]
                                for l in range(7):
                                        frets.append(r.readInt())
                                data["frets"]=frets
                                data["numBarres"]=r.readByte()[0]
                                barreList=[None,None,None,None,None]
                                for l in range(5):
                                        barreList[l]=[r.readByte()[0],None,None]
                                for l in range(5):
                                        barreList[l][0]=r.readByte()[0]
                                for l in range(5):
                                        barreList[l][1]=r.readByte()[0]
                                data["barres"]=barreList
                                ommissions=[]
                                for l in range(7):
                                        ommissions.append(r.readByte()[0])
                                data["ommissions"]=ommissions
                                r.skipBytes(1)
                                fingering=[]
                                for l in range(7):
                                        fingering.append(r.readByte()[0])
                                data["fingering"]=fingering
                                data["showFingering"]=r.readByte()[0]
                                return GPFile.GPBeat.GPChordDiagram(**data)

                        def write(self,w):
                                w.writeBytes(bytes([self.header]))
                                if self.sharp is None:
                                    w.skipBytes(16)
                                else:
                                    w.writeBytes(bytes([self.sharp]))
                                    w.skipBytes(3)
                                    w.writeBytes(bytes([self.root]))
                                    w.writeBytes(bytes([self.cType]))
                                    w.writeBytes(bytes([self.extra]))
                                    w.writeInt(self.bass)
                                    w.writeInt(self.dimaug)
                                    w.writeBytes(bytes([self.header]))
                                w.writeString(self.name)
                                w.skipBytes(20-len(self.name))
                                if self.sharp is None:
                                        w.skipBytes(5)
                                else:
                                        w.skipBytes(2)
                                        w.writeBytes(bytes([self.fifth]))
                                        w.writeBytes(bytes([self.ninth]))
                                        w.writeBytes(bytes([self.eleventh]))
                                w.writeInt(self.baseFret)
                                for f in self.frets:
                                        w.writeInt(f)
                                w.writeBytes(bytes([self.numBarres]))
                                for tuplePart in [0,1,2]:
                                    for b in self.barres:
                                        w.writeBytes(bytes([b[tuplePart]]))
                                for o in self.ommissions:
                                        w.writeBytes(bytes([o]))
                                w.skipBytes(1)
                                fingering=[]
                                for f in self.fingering:
                                        w.writeBytes(bytes([f]))
                                w.writeBytes(bytes([self.showFingering]))
                #END GPChordDiagram

                class GPBeatEffects:
                        def __init__(self, upStroke=None, downStroke=None, tapping=None, tremolo=None, pickstroke=None, rasguedo=None):
                                self.upStroke=upStroke
                                self.downStroke=downStroke
                                self.tapping=tapping
                                self.tremolo=tremolo
                                self.pickstroke=pickstroke
                                self.rasguedo=rasguedo

                        @staticmethod
                        def read(r, version):
                                eHeader=r.readByte()[0]
                                hasStroke=(eHeader & 0b1000000) > 0
                                hasTapping=(eHeader & 0b100000) > 0

                                hasTremolo=False
                                hasPickstroke=False
                                hasRasguedo=False
                                if "3.0" not in version:
                                        eHeader2=r.readByte()[0]
                                        hasTremolo=(eHeader2 & 0b100) > 0
                                        hasPickstroke=(eHeader2 & 0b10) > 0
                                        hasRasguedo=(eHeader2 & 0b1) > 0

                                data={}
                                if hasTapping:
                                        data["tapping"]=r.readByte()[0]

                                if hasTremolo:
                                        bType=r.readByte()[0]
                                        bHeight=r.readInt()
                                        numPoints=r.readInt()
                                        points=[]
                                        for p in range(numPoints):
                                                time=r.readInt()
                                                vPos=r.readInt()
                                                vibrato=r.readByte()[0]
                                                points.append((time,vPos,vibrato))
                                        data["tremolo"]=(bType,bHeight,points)

                                if hasStroke:
                                        data["upStroke"]=r.readByte()[0]
                                        data["downStroke"]=r.readByte()[0]

                                if hasRasguedo:
                                        data["rasguedo"]=r.readByte()[0]

                                if hasPickstroke:
                                        data["pickstroke"]=r.readByte()[0]
                                return GPFile.GPBeat.GPBeatEffects(**data)

                        def write(self, w):
                                w.writeMask([False, \
                                                False, \
                                                False, \
                                                False, \
                                                False, \
                                                self.tapping is not None, \
                                                self.upStroke is not None, \
                                                False])

                                w.writeMask([self.rasguedo is not None, \
                                                self.pickstroke is not None, \
                                                self.tremolo is not None, \
                                                False, \
                                                False, \
                                                False, \
                                                False, \
                                                False])

                                if self.tapping is not None:
                                        w.writeBytes(bytes([self.tapping]))

                                if self.tremolo is not None:
                                        (bType,bHeight,points)=self.tremolo
                                        w.writeBytes(bytes([bType]))
                                        w.writeInt(bHeight)
                                        w.writeInt(len(points))
                                        for p in points:
                                                (time,vPos,vibrato)=p
                                                w.writeInt(time)
                                                w.writeInt(vPos)
                                                w.writeBytes(bytes([vibrato]))

                                if self.upStroke is not None:
                                        w.writeBytes(bytes([self.upStroke]))
                                        w.writeBytes(bytes([self.downStroke]))

                                if self.rasguedo is not None:
                                        w.writeBytes(bytes([self.rasguedo]))

                                if self.pickstroke is not None:
                                        w.writeBytes(bytes([self.pickstroke]))

                #END GPBeatEffects

                class GPMixTableChange:
                        def __init__(self, instrument, allTracks, volume=None, pan=None, chorus=None, reverb=None, phaser=None, tremolo=None, tempo=None):
                                self.instrument=instrument
                                self.allTracks=allTracks
                                self.volume=volume
                                self.pan=pan
                                self.chorus=chorus
                                self.reverb=reverb
                                self.phaser=phaser
                                self.tremolo=tremolo
                                self.tempo=tempo

                        @staticmethod
                        def read(r,version):
                                data={}
                                data["instrument"]=r.readByte()[0]
                                volume=r.readByte()[0]
                                pan=r.readByte()[0]
                                chorus=r.readByte()[0]
                                reverb=r.readByte()[0]
                                phaser=r.readByte()[0]
                                tremolo=r.readByte()[0]
                                tempo=r.readInt(signed=True)
                                if volume!=0xff and "3.0" not in version:
                                        data["volume"]=(volume,r.readByte()[0])
                                if pan!=0xff and "3.0" not in version:
                                        data["pan"]=(pan,r.readByte()[0])
                                if chorus!=0xff:
                                        data["chorus"]=(chorus,r.readByte()[0])
                                if reverb!=0xff:
                                        data["reverb"]=(reverb,r.readByte()[0])
                                if phaser!=0xff:
                                        data["phaser"]=(phaser,r.readByte()[0])
                                if tremolo!=0xff:
                                        data["tremolo"]=(tremolo,r.readByte()[0])
                                if tempo!=-1:
                                        data["tempo"]=(tempo,r.readByte()[0])
                                data["allTracks"]=r.readByte()[0]
                                #print(data)
                                return GPFile.GPBeat.GPMixTableChange(**data)

                        def write(self,w):
                                w.writeBytes(bytes([self.instrument]))
                                for v in [self.volume, self.pan, self.chorus, self.reverb, self.phaser, self.tremolo]:
                                    if v is not None:
                                        w.writeBytes(bytes([v[0]]))
                                    else:
                                        w.writeBytes(bytes([0xff]))
                                if self.tempo is not None:
                                    w.writeInt(self.tempo[0], signed=True)
                                else:
                                    w.writeInt(-1, signed=True)
                                for v in [self.volume, self.pan, self.chorus, self.reverb, self.phaser, self.tremolo]:
                                    if v is not None:
                                        w.writeBytes(bytes([v[1]]))
                                if self.tempo is not None:
                                    w.writeBytes(bytes([self.tempo[1]]))
                                w.writeBytes(bytes([self.allTracks]))

                #END GPMixTableChange

                class GPNote:

                        class GPNoteEffects:
                                def __init__(self, letRing, hasHammerFrom, leftHandVibrato=None, palmMute=None, staccato=None, graceNote=None, slideFrom=None, bend=None, trill=None, harmonic=None, tremoloPicking=None):
                                        self.letRing=letRing
                                        self.hasHammerFrom=hasHammerFrom
                                        self.leftHandVibrato=leftHandVibrato
                                        self.palmMute=palmMute
                                        self.staccato=staccato
                                        self.graceNote=graceNote
                                        self.slideFrom=slideFrom
                                        self.bend=bend
                                        self.trill=trill
                                        self.harmonic=harmonic
                                        self.tremoloPicking=tremoloPicking

                                @staticmethod
                                def read(r, version):
                                        data={}
                                        neHeader=r.readByte()[0]
                                        hasGraceNote=(neHeader & 0b10000) > 0
                                        data["letRing"]=(neHeader & 0b1000) > 0
                                        hasSlideFrom=(neHeader & 0b100) > 0
                                        data["hasHammerFrom"]=(neHeader & 0b10) > 0
                                        hasBend=(neHeader & 0b1) > 0

                                        trill=False
                                        harmonic=False
                                        hasSlideFrom2=False
                                        tremoloPicking=False


                                        if "3.0" not in version:
                                                neHeader2=r.readByte()[0]
                                                data["leftHandVibrato"]=(neHeader2 & 0b1000000) > 0
                                                trill=(neHeader2 & 0b100000) > 0
                                                harmonic=(neHeader2 & 0b10000) > 0
                                                hasSlideFrom2=(neHeader2 & 0b1000) > 0
                                                tremoloPicking=(neHeader2 & 0b100) > 0
                                                data["palmMute"]=(neHeader2 & 0b10) > 0
                                                data["staccato"]=(neHeader2 & 0b1) > 0

                                        if hasBend:
                                                bType=r.readByte()[0]
                                                bHeight=r.readInt()
                                                numPoints=r.readInt()
                                                points=[]
                                                for p in range(numPoints):
                                                        time=r.readInt()
                                                        vPos=r.readInt()
                                                        vibrato=r.readByte()[0]
                                                        points.append((time,vPos,vibrato))
                                                data["bend"]=(bType,bHeight,points)

                                        if hasGraceNote:
                                                fret=r.readByte()[0]
                                                dynamic=r.readByte()[0]
                                                transition=r.readByte()[0]
                                                duration=r.readByte()[0]
                                                data["graceNote"]=(fret,dynamic,transition,duration)

                                        if tremoloPicking:
                                                data["tremoloPicking"]=r.readByte()[0]

                                        if hasSlideFrom2:
                                                data["slideFrom"]=r.readByte()[0]

                                        if harmonic:
                                                data["harmonic"]=r.readByte()[0]

                                        if trill:
                                                trillFret=r.readByte()[0]
                                                trillPeriod=r.readByte()[0]
                                                data["trill"]=(trillFret,trillPeriod)

                                        return GPFile.GPBeat.GPNote.GPNoteEffects(**data)
                                def write(self, w):
                                        w.writeMask([self.bend is not None, \
                                                        self.hasHammerFrom, \
                                                        self.slideFrom is not None, \
                                                        self.letRing, \
                                                        self.graceNote is not None, \
                                                        False, \
                                                        False, \
                                                        False])

                                        w.writeMask([self.staccato, \
                                                        self.palmMute, \
                                                        self.tremoloPicking is not None, \
                                                        self.slideFrom is not None, \
                                                        self.harmonic is not None, \
                                                        self.trill is not None, \
                                                        self.leftHandVibrato, \
                                                        False])

                                        if self.bend is not None:
                                                (bType,bHeight,points)=self.bend
                                                w.writeBytes(bytes([bType]))
                                                w.writeInt(bHeight)
                                                w.writeInt(len(points))
                                                for p in points:
                                                        (time,vPos,vibrato)=p
                                                        w.writeInt(time)
                                                        w.writeInt(vPos)
                                                        w.writeBytes(bytes([vibrato]))

                                        if self.graceNote is not None:
                                                (fret,dynamic,transition,duration)=self.graceNote
                                                w.writeBytes(bytes([fret]))
                                                w.writeBytes(bytes([dynamic]))
                                                w.writeBytes(bytes([transition]))
                                                w.writeBytes(bytes([duration]))

                                        if self.tremoloPicking is not None:
                                                w.writeBytes(bytes([self.tremoloPicking]))

                                        if self.slideFrom is not None:
                                                w.writeBytes(bytes([self.slideFrom]))

                                        if self.harmonic is not None:
                                                w.writeBytes(bytes([self.harmonic]))

                                        if self.trill is not None:
                                                (trillFret,trillPeriod)=self.trill
                                                w.writeBytes(bytes([trillFret]))
                                                w.writeBytes(bytes([trillPeriod]))

                        #END GPNoteEffects
                        def __init__(self, strength, isAccentuated, isGhost, isDotted, fingering=None, noteType=None, duration=None, ntuplet=None, effects=None):
                                self.strength=strength
                                self.isAccentuated=isAccentuated
                                self.isGhost=isGhost
                                self.isDotted=isDotted
                                self.fingering=fingering
                                self.noteType=noteType
                                self.duration=duration
                                self.ntuplet=ntuplet
                                self.effects=effects

                        @staticmethod
                        def read(r, version):
                                data={}
                                nHeader=r.readByte()[0]
                                right=(nHeader & 0b10000000) > 0
                                data["isAccentuated"]=(nHeader & 0b1000000) > 0
                                hasNoteType=(nHeader & 0b100000) > 0
                                hasNoteDynamic=(nHeader & 0b10000) > 0
                                hasEffects=(nHeader & 0b1000) > 0
                                data["isGhost"]=(nHeader & 0b100) > 0
                                data["isDotted"]=(nHeader & 0b10) > 0
                                hasDuration=(nHeader & 0b1) > 0


                                noteType=None
                                if hasNoteType:
                                        noteType=r.readByte()[0]
                                        #r.skipBytes(1)

                                if hasDuration:
                                        data["duration"]=r.readByte()[0]
                                        data["ntuplet"]=r.readByte()[0]

                                data["strength"]=6
                                if hasNoteDynamic:
                                        data["strength"]=r.readByte()[0]

                                if hasNoteType:
                                        data["noteType"]=(noteType,r.readByte()[0])

                                if right:
                                        data["fingering"]=(r.readByte()[0],r.readByte()[0])

                                if hasEffects:
                                        data["effects"]=GPFile.GPBeat.GPNote.GPNoteEffects.read(r, version)

                                return GPFile.GPBeat.GPNote(**data)

                        def write(self, w):
                                w.writeMask([self.duration is not None, \
                                                self.isDotted, \
                                                self.isGhost, \
                                                self.effects is not None, \
                                                self.strength is not None, \
                                                self.noteType is not None, \
                                                self.isAccentuated, \
                                                self.fingering is not None])

                                if self.noteType is not None:
                                        w.writeBytes(bytes([self.noteType[0]]))

                                if self.duration is not None:
                                        w.writeBytes(bytes([self.duration]))
                                        w.writeBytes(bytes([self.ntuplet]))

                                if self.strength is not None:
                                        w.writeBytes(bytes([self.strength]))

                                if self.noteType is not None:
                                        w.writeBytes(bytes([self.noteType[1]]))

                                if self.fingering is not None:
                                        w.writeBytes(bytes([self.fingering[0]]))
                                        w.writeBytes(bytes([self.fingering[1]]))

                                if self.effects is not None:
                                        self.effects.write(w)

                #END GPNote

                def __init__(self, duration=0, strings=[None,None,None,None,None,None,None,None], status=None, ntuplet=None, mixTableChange=None, effects=None, text=None, chordDiagram=None, isDotted=False):
                        self.duration=duration
                        self.strings=strings
                        self.status=status
                        self.ntuplet=ntuplet
                        self.mixTableChange=mixTableChange
                        self.effects=effects
                        self.text=text
                        self.chordDiagram=chordDiagram
                        self.isDotted=isDotted

                def getSummary(self):
                        result=""
                        for s in self.strings[1:-1]:
                                if not s:
                                        result+="  |"
                                else:
                                        result+=(" "+str(s.noteType[1]))[-2:]+"|"
                        return result

                @staticmethod
                def read(r, version):
                        data={}
                        bHeader=r.readByte()[0]
                        hasStatus=(bHeader & 0b1000000) > 0
                        ntuplet=(bHeader & 0b100000) > 0
                        mixTableChange=(bHeader & 0b10000) > 0
                        hasEffects=(bHeader & 0b1000) > 0
                        hasText=(bHeader & 0b100) > 0
                        chordDiagram=(bHeader & 0b10) > 0
                        data["isDotted"]=(bHeader & 0b1) > 0

                        if hasStatus:
                                data["status"]=r.readByte()
                        data["duration"]=r.readByte()

                        if ntuplet:
                                data["ntuplet"]=r.readInt()

                        if chordDiagram:
                                data["chordDiagram"]=GPFile.GPBeat.GPChordDiagram.read(r,version)

                        if hasText:
                                int_size=r.readInt()
                                data["text"]=r.readString()

                        if hasEffects:
                                data["effects"]=GPFile.GPBeat.GPBeatEffects.read(r,version)

                        if mixTableChange:
                                data["mixTableChange"]=GPFile.GPBeat.GPMixTableChange.read(r,version)

                        #print("Reading strings")
                        #r.printPos()
                        involvedStrings=r.readByte()[0]
                        usesSixth=(involvedStrings & 0b1000000) > 0
                        usesFifth=(involvedStrings & 0b100000) > 0
                        usesFourth=(involvedStrings & 0b10000) > 0
                        usesThird=(involvedStrings & 0b1000) > 0
                        usesSecond=(involvedStrings & 0b100) > 0
                        usesFirst=(involvedStrings & 0b10) > 0
                        #print(involvedStrings)

                        notes=[None,None,None,None,None,None,None,None]
                        for pos,active in enumerate([False,usesSixth,usesFifth,usesFourth,usesThird,usesSecond,usesFirst]):
                                if active:
                                        notes[pos]=GPFile.GPBeat.GPNote.read(r,version)
                        data["strings"]=notes
                        return GPFile.GPBeat(**data)

                def write(self, w):
                        w.writeMask([self.isDotted, \
                                        self.chordDiagram is not None, \
                                        self.text is not None, \
                                        self.effects is not None, \
                                        self.mixTableChange is not None, \
                                        self.ntuplet is not None, \
                                        self.status is not None, \
                                        False])

                        if self.status is not None:
                                w.writeBytes(self.status)
                        w.writeBytes(self.duration)

                        if self.ntuplet is not None:
                                w.writeInt(self.ntuplet)

                        if self.chordDiagram is not None:
                                self.chordDiagram.write(w)

                        if self.text is not None:
                                w.writeInt(len(self.text)+1)
                                w.writeString(self.text)

                        if self.effects is not None:
                                self.effects.write(w)

                        if self.mixTableChange is not None:
                                self.mixTableChange.write(w)
                                
                        w.writeMask(list(map(lambda x: x is not None,reversed(self.strings))))
                        for s in self.strings:
                                if s is not None:
                                        s.write(w)

        #END GPBeat



        def __init__(self,header,measures,tracks,beatLists):
                self.header=header
                self.measures=measures
                self.tracks=tracks
                self.beatLists=beatLists

        def print(self):
                print("----GPHeader----")
                self.header.print(prefix='\t')
                print("----End GPHeader----")
                for i,m in enumerate(self.measures):
                        m.print("MEASURE %d: "%i)
                for i,t in enumerate(self.tracks):
                        t.print("TRACK %d: "%i)

                #DEBUG
                for i,t in enumerate(self.tracks):
                        print("\nLast measure for track %d"%i)
                        lastBeatList=self.beatLists[-4][i]
                        for b in lastBeatList:
                                print(b.getSummary())

        @staticmethod
        def read(filename):
                r=Reader(filename)
                header=GPFile.GPHeader.read(r)
                measures=[]
                for i in range(header.numMeasures):
                        m=GPFile.GPMeasure.read(r)
                        measures.append(m)
                tracks=[]
                for i in range(header.numTracks):
                        t=GPFile.GPTrack.read(r)
                        tracks.append(t)

                beatLists=[]
                #r.printPos()
                for i in range(header.numMeasures):
                        beatListsForMeasure=[]
                        for j in range(header.numTracks):
                                numBeats=r.readInt()
                                beats=[]
                                #print("NUMBEATS %d:"%numBeats)
                                for k in range(numBeats):
                                        beats.append(GPFile.GPBeat.read(r,header.version))
                                        #r.printPos()
                                        #print("M%d - T%d - %d/%d "%(i,j,k+1,numBeats) + beats[-1].getSummary())
                                beatListsForMeasure.append(beats)
                        beatLists.append(beatListsForMeasure)
                r.close()
                return GPFile(header, measures, tracks, beatLists)

        def write(self, filename):
                w=Writer(filename)
                self.header.write(w)
                for m in self.measures:
                        m.write(w)
                for t in self.tracks:
                        t.write(w)
                for beatListsForMeasure in self.beatLists:
                    for beatsForTrackForMeasure in beatListsForMeasure:
                        w.writeInt(len(beatsForTrackForMeasure))
                        for beat in beatsForTrackForMeasure:
                            beat.write(w)
                w.close()
                
        def dropAllTracksBut(self,num):
                self.header.numTracks=1
                self.tracks=[self.tracks[num]]
                for i in range(len(self.beatLists)):
                        self.beatLists[i]=[self.beatLists[i][num]]
                        
        def strip(self):
                firstValid=0
                found=False
                for i,beatListsForMeasure in enumerate(self.beatLists):
                        for beatsForTrackForMeasure in beatListsForMeasure:
                                for beat in beatsForTrackForMeasure:
                                        if beat.status is None:
                                                firstValid=i
                                                found=True
                                                break
                                if found:
                                        break
                        if found:
                                break
                                
                lastValid=len(self.beatLists)
                found=False
                for i,beatListsForMeasure in zip(itertools.count(len(self.beatLists) - 1, -1), reversed(self.beatLists)):
                        for beatsForTrackForMeasure in beatListsForMeasure:
                                for beat in beatsForTrackForMeasure:
                                        if beat.status is None:
                                                lastValid=i
                                                found=True
                                                break
                                if found:
                                        break
                        if found:
                                break
                
                self.measures=self.measures[firstValid:lastValid]
                self.beatLists=self.beatLists[firstValid:lastValid]
                self.header.numMeasures=len(self.beatLists)